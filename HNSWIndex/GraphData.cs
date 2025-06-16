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
    internal class GraphData<TLabel, TDistance> where TDistance : struct, IFloatingPoint<TDistance>
    {
        internal event EventHandler<ReallocateEventArgs>? Reallocated;

        internal object indexLock = new object();

        internal List<Node> Nodes { get; private set; }

        internal ConcurrentDictionary<int, TLabel> Items { get; private set; }

        private object removedIndexesLock = new object();

        internal Queue<int> RemovedIndexes { get; private set; }

        internal object NeighbourhoodBitmapLock = new object();

        internal List<bool> NeighbourhoodBitmap { get; private set; }

        internal object entryPointLock = new object();

        internal int EntryPointId = -1;

        internal Node EntryPoint => Nodes[EntryPointId];

        internal int Capacity;

        private object rngLock = new object();

        private Random rng;

        private double distRate;

        private int maxEdges;

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
            Capacity = parameters.CollectionSize;

            RemovedIndexes = new Queue<int>();
            Nodes = new List<Node>(parameters.CollectionSize);
            NeighbourhoodBitmap = new List<bool>(parameters.CollectionSize);
            Items = new ConcurrentDictionary<int, TLabel>(65536, parameters.CollectionSize); // 2^16 amount of locks
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

            Nodes = snapshot.Nodes ?? new List<Node>(parameters.CollectionSize);
            Items = snapshot.Items ?? new ConcurrentDictionary<int, TLabel>(65536, parameters.CollectionSize);
            RemovedIndexes = snapshot.RemovedIndexes ?? new Queue<int>();
            EntryPointId = snapshot.EntryPointId;
            Capacity = snapshot.Capacity;

            NeighbourhoodBitmap = new List<bool>(Capacity);
            for (int i = 0; i < Nodes.Count; i++)
                NeighbourhoodBitmap.Add(false);
        }

        /// <summary>
        /// Add new item to the graph.
        /// </summary>
        internal int AddItem(TLabel item)
        {
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
                Nodes[vacantId] = NewNode(vacantId);
                Items.TryAdd(vacantId, item);
                return vacantId;
            }

            vacantId = Nodes.Count;
            Nodes.Add(NewNode(vacantId));
            NeighbourhoodBitmap.Add(false);
            Items.TryAdd(vacantId, item);
            if (Nodes.Count > Capacity)
            {
                Capacity = Nodes.Capacity;
                Reallocated?.Invoke(this, new ReallocateEventArgs(Capacity));
            }
            return vacantId;
        }

        /// <summary>
        /// Remove item from lookup table and mark its index as free.
        /// It is still possible to access the node with this index, until new item takes its place.
        /// </summary>
        internal void RemoveItem(int itemId)
        {
            Items.TryRemove(itemId, out _);
            lock (removedIndexesLock)
            {
                RemovedIndexes.Enqueue(itemId);
            }
        }

        /// <summary>
        /// Move role of the entry point to another point the graph.
        /// </summary>
        internal void RemoveEntryPoint()
        {
            lock (entryPointLock)
            {
                for (int layer = GetTopLayer(); layer >= 0; layer--)
                {
                    if (EntryPoint.OutEdges[layer].Count > 0)
                    {
                        var neighbourId = EntryPoint.OutEdges[layer].MaxBy(id => Nodes[id].OutEdges.Count);
                        SetEntryPoint(neighbourId);
                        return;
                    }
                }
            }
        }

        /// <summary>
        /// Setter for entry point of the graph.
        /// </summary>
        internal void SetEntryPoint(int epId)
        {
            // TODO: Remove this method and write proper setter for EntryPointId.
            EntryPointId = epId;
        }

        /// <summary>
        /// Get the maximum layer of the graph.
        /// </summary>
        internal int GetTopLayer()
        {
            return Nodes[EntryPointId].MaxLayer;
        }

        /// <summary>
        /// Constriction function for new node structure.
        /// </summary>
        private Node NewNode(int index)
        {
            float random = 0;
            lock (rngLock)
            {
                random = rng.NextSingle();
            }

            int topLayer = (int)(-Math.Log(random) * distRate);

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
        /// Wait until all neighbours of the node at given layer are free
        /// Then lock all those neighbours until processing is done
        /// </summary>
        internal void LockNodeNeighbourhood(Node node, int layer)
        {
            lock (NeighbourhoodBitmapLock)
            {
                while (NeighbourhoodIsBusy(node, layer))
                {
                    Monitor.Wait(NeighbourhoodBitmapLock);
                }

                NeighbourhoodBitmap[node.Id] = true;
                foreach (var neighbourId in node.OutEdges[layer])
                    NeighbourhoodBitmap[neighbourId] = true;
                foreach (var neighbourId in node.InEdges[layer])
                    NeighbourhoodBitmap[neighbourId] = true;
            }
        }

        /// <summary>
        /// Free all nodes locked with LockNodeNeighbourhood method
        /// </summary>
        internal void UnlockNodeNeighbourhood(Node node, int layer)
        {
            lock (NeighbourhoodBitmapLock)
            {
                NeighbourhoodBitmap[node.Id] = false;
                foreach (var neighbourId in node.OutEdges[layer])
                    NeighbourhoodBitmap[neighbourId] = false;
                foreach (var neighbourId in node.InEdges[layer])
                    NeighbourhoodBitmap[neighbourId] = false;
                Monitor.PulseAll(NeighbourhoodBitmapLock);
            }
        }

        /// <summary>
        /// Check if the node and its neighbours are busy at given layer.
        /// This method is used to acquire lock for region of the graph associated with the node.
        /// </summary>
        internal bool NeighbourhoodIsBusy(Node node, int layer)
        {
            bool result = NeighbourhoodBitmap[node.Id];
            for (int i = 0; i < node.OutEdges[layer].Count; i++)
            {
                var neighbourId = node.OutEdges[layer][i];
                result |= NeighbourhoodBitmap[neighbourId];
            }
            for (int i = 0; i < node.InEdges[layer].Count; i++)
            {
                var neighbourId = node.InEdges[layer][i];
                result |= NeighbourhoodBitmap[neighbourId];
            }
            return result;
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
    }
}