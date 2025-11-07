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
    internal class GraphData<TLabel, TDistance> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance>
    {
        internal event EventHandler<ReallocateEventArgs>? Reallocated;

        internal object indexLock = new object();

        internal Node[] Nodes { get; private set; }

        internal TLabel[] Items { get; private set; }

        private object removedIndexesLock = new object();

        internal ConcurrentQueue<int> RemovedIndexes { get; private set; }


        internal GraphRegionLocker GraphLocker;

        internal object entryPointLock = new object();

        internal int EntryPointId = -1;

        internal Node EntryPoint => Nodes[EntryPointId];

        internal int Capacity;

        internal int Length = 0;

        internal int Count = 0;

        private Random rng;

        private double distRate;

        private int maxEdges;

        private bool zeroLayerGuaranteed;

        private bool allowRemovals;

        private Func<TLabel, TLabel, TDistance> distanceFnc;

        /// <summary>
        /// Constructor for the graph data.
        /// </summary>
        internal GraphData(Func<TLabel, TLabel, TDistance> distance, HNSWParameters<TDistance> parameters)
        {
            distanceFnc = distance;
            rng = parameters.RandomSeed < 0 ? new Random() : new Random(parameters.RandomSeed);
            distRate = parameters.DistributionRate;
            maxEdges = parameters.MaxEdges;
            zeroLayerGuaranteed = parameters.ZeroLayerGuaranteed;
            allowRemovals = parameters.AllowRemovals;
            Capacity = parameters.CollectionSize;

            RemovedIndexes = new ConcurrentQueue<int>();
            Nodes = new Node[parameters.CollectionSize];
            Items = new TLabel[parameters.CollectionSize];
            GraphLocker = new GraphRegionLocker(parameters.CollectionSize);
        }

        /// <summary>
        /// Constructor for the graph data from serialization snapshot.
        /// </summary>
        internal GraphData(GraphDataSnapshot<TLabel, TDistance> snapshot, Func<TLabel, TLabel, TDistance> distance, HNSWParameters<TDistance> parameters)
        {
            distanceFnc = distance;
            rng = parameters.RandomSeed < 0 ? new Random() : new Random(parameters.RandomSeed);
            distRate = parameters.DistributionRate;
            maxEdges = parameters.MaxEdges;
            zeroLayerGuaranteed = parameters.ZeroLayerGuaranteed;
            allowRemovals = parameters.AllowRemovals;

            Nodes = snapshot.ParsedNodes ?? new Node[parameters.CollectionSize];
            Items = snapshot.ParsedItems ?? new TLabel[parameters.CollectionSize];
            GraphLocker = new GraphRegionLocker(snapshot.Capacity);
            RemovedIndexes = snapshot.RemovedIndexes ?? new ConcurrentQueue<int>();
            EntryPointId = snapshot.EntryPointId;
            Capacity = snapshot.Capacity;
            Length = snapshot.Length;
            Count = snapshot.Count;
        }

        /// <summary>
        /// Add new item to the graph.
        /// </summary>
        internal int AddItem(TLabel item)
        {
            var topLayer = GetRandomLayer();
            if (topLayer < 0) return -1;

            // Search for empty spot first
            if (allowRemovals && RemovedIndexes.TryDequeue(out int vacantId))
            {
                Nodes[vacantId] = NewNode(vacantId, topLayer);
                Items[vacantId] = item;
                Interlocked.Increment(ref Count);
                return vacantId;
            }

            // Allocate new spot
            var len = Interlocked.Increment(ref Length);
            lock (indexLock)
            {
                if (len >= Capacity)
                {
                    Capacity *= 2;
                    var nodes = Nodes;
                    var items = Items;
                    Array.Resize(ref nodes, Capacity);
                    Array.Resize(ref items, Capacity);
                    Nodes = nodes;
                    Items = items;
                    // Update other structures
                    Reallocated?.Invoke(this, new ReallocateEventArgs(Capacity));
                    GraphLocker.UpdateCapacity(Capacity);
                }
                vacantId = len - 1;
                Nodes[vacantId] = NewNode(vacantId, topLayer);
                Items[vacantId] = item;
            }

            Interlocked.Increment(ref Count);
            return vacantId;
        }

        /// <summary>
        /// Remove item from lookup table and mark its index as free.
        /// It is still possible to access the node with this index, until new item takes its place.
        /// </summary>
        internal void RemoveItem(int itemId)
        {
            Items[itemId] = default!;
            lock (removedIndexesLock)
            {
                RemovedIndexes.Enqueue(itemId);
                Count--;
            }
        }

        /// <summary>
        /// Replace node at given id
        /// </summary>
        internal int UpdateItem(int itemId, TLabel label)
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
                int neighbourId = -1;
                int maxConnections = 0;
                for (int i = 0; i < EntryPoint.OutEdges[layer].Count; i++)
                {
                    var neighborId = EntryPoint.OutEdges[layer].AsSpan()[i];
                    var neighbor = Nodes[neighborId];
                    if (neighbor.OutEdges[layer].Count > maxConnections)
                    {
                        maxConnections = neighbor.OutEdges[layer].Count;
                        neighbourId = neighborId;
                    }
                }
                EntryPointId = neighbourId;
                return true;
            }
            return false;
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
            float random = Random.Shared.NextSingle();
            return zeroLayerGuaranteed ? (int)(-Math.Log(random) * distRate) : (int)(-Math.Log(random) * distRate) - 1;
        }

        /// <summary>
        /// Constriction function for new node structure.
        /// </summary>
        private Node NewNode(int index, int topLayer)
        {
            var outEdges = new EdgeList[topLayer + 1];
            var inEdges = new EdgeList[topLayer + 1];

            for (int layer = 0; layer <= topLayer; layer++)
            {
                int maxEdges = MaxEdges(layer);
                outEdges[layer] = new EdgeList(maxEdges + 1);
                inEdges[layer] = new EdgeList(allowRemovals ? maxEdges + 1 : 0);
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
        internal TDistance Distance(TLabel a, TLabel b)
        {
            return distanceFnc(a, b);
        }

        /// <summary>
        /// Proxy for distance between graph vertex and arbitrary point
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal TDistance Distance(int a, TLabel b)
        {
            return distanceFnc(Items[a], b);
        }
    }
}