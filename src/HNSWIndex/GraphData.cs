using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace HNSWIndex
{
    /// <summary>
    /// Class storing the data containers for HNSW index.
    /// All lock related members are ommitted from serialization 
    /// and should be initialized in deserialization constructor.
    /// </summary>
    internal class GraphData<TVector, TDistance> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance>
    {
        internal event EventHandler<ReallocateEventArgs>? Reallocated;

        internal object indexLock = new object();
        internal Node[] Nodes { get; private set; }
        internal TVector[] Items { get; private set; }
        internal ConcurrentStack<int> RemovedIndexes { get; private set; }
        internal GraphRegionLocker GraphLocker;
        internal object entryPointLock = new object();
        internal int EntryPointId = -1;
        internal Node EntryPoint => Nodes[EntryPointId];
        internal int Capacity;
        internal int Length = 0;
        internal int Count => activeNodes.Count;
        internal int[] ActiveIds => activeNodes.ActiveIds;
        private ActiveSet activeNodes;
        private object rngLock = new object();
        private Random rng;
        private double distRate;
        private int maxEdges;
        private bool allowRemovals;
        private Func<TVector, TVector, TDistance> distanceFnc;

        /// <summary>
        /// Constructor for the graph data.
        /// </summary>
        internal GraphData(Func<TVector, TVector, TDistance> distance, HNSWParameters<TDistance> parameters)
        {
            distanceFnc = distance;
            rng = parameters.RandomSeed < 0 ? new Random() : new Random(parameters.RandomSeed);
            distRate = parameters.DistributionRate;
            maxEdges = parameters.MaxEdges;
            allowRemovals = parameters.AllowRemovals;
            Capacity = parameters.CollectionSize;

            RemovedIndexes = new ConcurrentStack<int>();
            Nodes = new Node[parameters.CollectionSize];
            Items = new TVector[parameters.CollectionSize];
            activeNodes = new ActiveSet(parameters.CollectionSize);
            GraphLocker = new GraphRegionLocker(parameters.CollectionSize);
        }

        /// <summary>
        /// Constructor for the graph data from serialization snapshot.
        /// </summary>
        internal GraphData(GraphDataSnapshot<TVector, TDistance> snapshot, Func<TVector, TVector, TDistance> distance, HNSWParameters<TDistance> parameters)
        {
            distanceFnc = distance;
            rng = parameters.RandomSeed < 0 ? new Random() : new Random(parameters.RandomSeed);
            distRate = parameters.DistributionRate;
            maxEdges = parameters.MaxEdges;
            allowRemovals = parameters.AllowRemovals;

            Nodes = snapshot.ParsedNodes ?? new Node[parameters.CollectionSize];
            Items = snapshot.ParsedItems ?? new TVector[parameters.CollectionSize];
            activeNodes = new ActiveSet(snapshot.ActiveNodes ?? new int[0]);
            GraphLocker = new GraphRegionLocker(snapshot.Capacity);
            RemovedIndexes = snapshot.RemovedIndexes ?? new ConcurrentStack<int>();
            EntryPointId = snapshot.EntryPointId;
            Capacity = snapshot.Capacity;
            Length = snapshot.Length;
        }

        /// <summary>
        /// Add new item to the graph.
        /// </summary>
        internal int AddItem(TVector item)
        {
            var topLayer = GetRandomLayer();
            if (topLayer < 0) return -1;

            // Search for empty spot first
            if (allowRemovals && RemovedIndexes.TryPop(out int vacantId))
            {
                Nodes[vacantId] = NewNode(vacantId, topLayer);
                Items[vacantId] = item;
                activeNodes.Add(vacantId);
                return vacantId;
            }

            // Allocate new spot
            int slotId;
            lock (indexLock)
            {
                slotId = Length++;
                if (Length > Capacity)
                {
                    Capacity *= 2;
                    var nodes = Nodes;
                    var items = Items;
                    Array.Resize(ref nodes, Capacity);
                    Array.Resize(ref items, Capacity);
                    Nodes = nodes;
                    Items = items;
                    // Update other structures
                    activeNodes.EnsureCapacity(Capacity);
                    Reallocated?.Invoke(this, new ReallocateEventArgs(Capacity));
                    GraphLocker.UpdateCapacity(Capacity);
                }
                Nodes[slotId] = NewNode(slotId, topLayer);
                Items[slotId] = item;
                activeNodes.Add(slotId);
            }

            return slotId;
        }

        /// <summary>
        /// Remove item from lookup table and mark its index as free.
        /// It is still possible to access the node with this index, until new item takes its place.
        /// </summary>
        internal void RemoveItem(int itemId)
        {
            RemovedIndexes.Push(itemId);
            activeNodes.Remove(itemId);
        }

        /// <summary>
        /// Replace node at given id
        /// </summary>
        internal int UpdateItem(int itemId, TVector label)
        {
            var topLayer = GetRandomLayer();
            if (topLayer < 0) return -1;
            Nodes[itemId] = NewNode(itemId, topLayer);
            Items[itemId] = label;
            return itemId;
        }

        /// <summary>
        /// Try to move the role of entry point to neighbor at given layer.
        /// This operations should be performed under neighborhhod lock of EP.
        /// </summary>
        internal bool TryReplaceEntryPoint(int layer)
        {
            if (EntryPoint.OutEdges[layer].Count > 0)
            {
                int replacementId = -1;
                int maxConnections = -1;
                for (int i = 0; i < EntryPoint.OutEdges[layer].Count; i++)
                {
                    var neighborId = EntryPoint.OutEdges[layer].AsSpan()[i];
                    var neighbor = Nodes[neighborId];
                    if (neighbor.OutEdges[layer].Count > maxConnections)
                    {
                        maxConnections = neighbor.OutEdges[layer].Count;
                        replacementId = neighborId;
                    }
                }
                EntryPointId = replacementId;
                return true;
            }
            return false;
        }

        /// <summary>
        /// Force replace entry point with point at highest layer.
        /// This requires scan through all active nodes.
        /// </summary>
        internal void ForceReplaceEntryPoint()
        {
            if (Count == 0) return;

            int bestLayer = -1;
            int bestId = -1;
            var activeIds = ActiveIds;
            for (int i = 0; i < Count; i++)
            {
                var candidate = Nodes[activeIds[i]];
                if (candidate.MaxLayer > bestLayer)
                {
                    bestLayer = candidate.MaxLayer;
                    bestId = candidate.Id;
                }
            }
            EntryPointId = bestId;
        }

        /// <summary>
        /// Get the maximum layer of the graph.
        /// </summary>
        internal int GetTopLayer()
        {
            return Nodes[EntryPointId].MaxLayer;
        }

        /// <summary>
        /// Take random layer based on parameter's distribution rate.
        /// If ZeroLayerGuaranteed flag is set then all points should be at least at layer zero.
        /// </summary>
        private int GetRandomLayer()
        {
            float random;
            lock (rngLock)
            {
                random = rng.NextSingle();
            }
            return (int)(-Math.Log(random) * distRate);
        }

        /// <summary>
        /// Constriction function for new node structure.
        /// </summary>
        private Node NewNode(int index, int topLayer)
        {
            var outEdges = new EdgeList[topLayer + 1];
            var inEdges = allowRemovals ? new EdgeList[topLayer + 1] : [];

            for (int layer = 0; layer <= topLayer; layer++)
            {
                int maxEdges = MaxEdges(layer);
                outEdges[layer] = new EdgeList(maxEdges + 1);
                if (allowRemovals) inEdges[layer] = new EdgeList(allowRemovals ? maxEdges + 1 : 0);
            }

            return new Node
            {
                Id = index,
                OutEdges = outEdges,
                InEdges = inEdges,
            };
        }

        /// <summary>
        /// Get maximum number of edges at given layer.
        /// </summary>
        internal int MaxEdges(int layer)
        {
            return layer == 0 ? maxEdges * 2 : maxEdges;
        }

        /// <summary>
        /// Wrapper for distance function working on indexes.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal TDistance Distance(int a, int b)
        {
            return distanceFnc(Items[a], Items[b]);
        }

        /// <summary>
        /// Proxy for distance function
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal TDistance Distance(TVector a, TVector b)
        {
            return distanceFnc(a, b);
        }

        /// <summary>
        /// Proxy for distance between graph vertex and arbitrary point
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal TDistance Distance(int a, TVector b)
        {
            return distanceFnc(Items[a], b);
        }
    }
}