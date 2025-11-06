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

        internal Queue<int> RemovedIndexes { get; private set; }


        internal GraphRegionLocker GraphLocker;

        internal object entryPointLock = new object();

        internal int EntryPointId = -1;

        internal Node EntryPoint => Nodes[EntryPointId];

        internal int Capacity;

        internal int Length = 0;

        internal int Count = 0;

        private object rngLock = new object();

        private Random rng;

        private double distRate;

        private int maxEdges;

        private bool zeroLayerGuaranteed;

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
            Capacity = parameters.CollectionSize;

            RemovedIndexes = new Queue<int>();
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

            Nodes = snapshot.ParsedNodes ?? new Node[parameters.CollectionSize];
            Items = snapshot.ParsedItems ?? new TLabel[parameters.CollectionSize];
            GraphLocker = new GraphRegionLocker(snapshot.Capacity);
            RemovedIndexes = snapshot.RemovedIndexes ?? new Queue<int>();
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
            int vacantId = -1;
            lock (removedIndexesLock)
            {
                if (RemovedIndexes.Count > 0)
                {
                    vacantId = RemovedIndexes.Dequeue();
                }
            }

            if (vacantId >= 0)
            {
                Nodes[vacantId] = NewNode(vacantId, topLayer);
                Items[vacantId] = item;
                Count++;
                return vacantId;
            }

            // Allocate new spot
            vacantId = Length++;
            Nodes[vacantId] = NewNode(vacantId, topLayer);
            Items[vacantId] = item;
            if (Length >= Capacity)
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
            Count++;
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
                var neighbourId = EntryPoint.OutEdges[layer].MaxBy(id => Nodes[id].OutEdges[layer].Count);
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
            float random = 0;
            lock (rngLock)
            {
                random = rng.NextSingle();
            }
            return zeroLayerGuaranteed ? (int)(-Math.Log(random) * distRate) : (int)(-Math.Log(random) * distRate) - 1;
        }

        /// <summary>
        /// Constriction function for new node structure.
        /// </summary>
        private Node NewNode(int index, int topLayer)
        {
            var outEdges = new List<List<int>>(topLayer + 1);
            var inEdges = new List<List<int>>(topLayer + 1);

            for (int layer = 0; layer <= topLayer; layer++)
            {
                int maxEdges = MaxEdges(layer);
                outEdges.Add(new List<int>(maxEdges + 1));
                inEdges.Add(new List<int>(maxEdges + 1));
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