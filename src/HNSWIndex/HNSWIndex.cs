using System.Numerics;
using ProtoBuf;

namespace HNSWIndex
{
    public class HNSWIndex<TVector, TDistance> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance>
    {
        private readonly HNSWParameters<TDistance> parameters;

        private readonly GraphData<TVector, TDistance> data;

        private readonly GraphConnector<TVector, TDistance> connector;

        private readonly GraphNavigator<TVector, TDistance> navigator;


        /// <summary>
        /// Construct KNN search graph with arbitrary distance function
        /// </summary>
        public HNSWIndex(Func<TVector, TVector, TDistance> distFnc, HNSWParameters<TDistance>? hnswParameters = null)
        {
            hnswParameters ??= new HNSWParameters<TDistance>();
            parameters = hnswParameters;

            data = new GraphData<TVector, TDistance>(distFnc, hnswParameters);
            navigator = new GraphNavigator<TVector, TDistance>(data);
            connector = new GraphConnector<TVector, TDistance>(data, navigator, hnswParameters);

            data.Reallocated += OnDataResized;
        }

        /// <summary>
        /// Construct KNN search graph from serialized snapshot.
        /// </summary>
        internal HNSWIndex(Func<TVector, TVector, TDistance> distFnc, HNSWIndexSnapshot<TVector, TDistance> snapshot)
        {
            if (snapshot.Parameters is null)
                throw new ArgumentNullException(nameof(snapshot.Parameters), "Parameters cannot be null during deserialization.");

            if (snapshot.DataSnapshot is null)
                throw new ArgumentNullException(nameof(snapshot.DataSnapshot), "Data cannot be null during deserialization.");

            parameters = snapshot.Parameters;
            data = new GraphData<TVector, TDistance>(snapshot.DataSnapshot, distFnc, snapshot.Parameters);

            navigator = new GraphNavigator<TVector, TDistance>(data);
            connector = new GraphConnector<TVector, TDistance>(data, navigator, parameters);

            data.Reallocated += OnDataResized;
        }

        /// <summary>
        /// Add new item with given label to the graph.
        /// </summary>
        public int Add(TVector item)
        {
            var itemId = data.AddItem(item);
            if (itemId == -1) return itemId;

            lock (data.Nodes[itemId].OutEdgesLock)
            {
                connector.ConnectNewNode(itemId);
            }
            return itemId;
        }

        /// <summary>
        /// Add collection of items to the graph
        /// </summary>
        public int[] Add(List<TVector> items)
        {
            var idArray = new int[items.Count];
            Parallel.For(0, items.Count, (i) =>
            {
                idArray[i] = Add(items[i]);
            });
            return idArray;
        }

        /// <summary>
        /// Remove item with given index from graph structure
        /// </summary>
        public void Remove(int itemIndex)
        {
            if (parameters.AllowRemovals == false)
                throw new InvalidOperationException("Removals are disabled in this index instance.");
            var item = data.Nodes[itemIndex];
            connector.RemoveNodeConnections(item);
        }

        /// <summary>
        /// Remove collection of items associated with indexes
        /// </summary>
        public void Remove(List<int> indexes)
        {
            Parallel.For(0, indexes.Count, (i) =>
            {
                Remove(indexes[i]);
            });
        }

        /// <summary>
        /// Get K nearest neighbors of query point. 
        /// Optionally provide filter function to ignore certain labels.
        /// Layer parameters indicates at which layer search should be performed (0 - base layer)
        /// </summary>
        public List<KNNResult<TVector, TDistance>> KnnQuery(TVector query, int k, Func<TVector, bool>? filterFnc = null, int layer = 0)
        {
            if (data.Count <= 0 || k < 1) return new List<KNNResult<TVector, TDistance>>();

            Func<int, bool> indexFilter = _ => true;
            if (filterFnc is not null)
                indexFilter = (index) => filterFnc(data.Items[index]);

            var neighborsAmount = Math.Max(parameters.MinNN, k);
            var ep = navigator.FindEntryPointQuery(layer, query);
            var topCandidates = navigator.SearchLayerQuery(ep.Id, layer, neighborsAmount, query, indexFilter);

            if (k < neighborsAmount)
            {
                return topCandidates.OrderBy(c => c.Dist).Take(k).ToList().ConvertAll(CandidateToResult);
            }
            return topCandidates.OrderBy(c => c.Dist).ToList().ConvertAll(CandidateToResult);
        }

        /// <summary>
        /// Perform batch knn query.
        /// </summary>
        public List<KNNResult<TVector, TDistance>>[] BatchKnnQuery(IList<TVector> queries, int k, Func<TVector, bool>? filterFnc = null, int layer = 0)
        {
            var result = new List<KNNResult<TVector, TDistance>>[queries.Count];
            Parallel.For(0, queries.Count, (i) =>
            {
                result[i] = KnnQuery(queries[i], k, filterFnc, layer);
            });
            return result;
        }

        /// <summary>
        /// Get all neighbors of query point which are within range distance.
        /// Optionally provide filter function to ignore certain labels.
        /// Layer parameters indicates at which layer search should be performed (0 - base layer)
        /// </summary>
        public List<KNNResult<TVector, TDistance>> RangeQuery(TVector query, TDistance range, Func<TVector, bool>? filterFnc = null, int layer = 0)
        {
            if (data.Count <= 0) return new List<KNNResult<TVector, TDistance>>();

            Func<int, bool> indexFilter = _ => true;
            if (filterFnc is not null)
                indexFilter = (index) => filterFnc(data.Items[index]);

            var ep = navigator.FindEntryPointQuery(layer, query);
            var topCandidates = navigator.SearchLayerRange(ep.Id, layer, range, query, indexFilter);
            return topCandidates.OrderBy(c => c.Dist).ToList().ConvertAll(CandidateToResult);
        }

        /// <summary>
        /// Perform batch range query.
        /// </summary>
        public List<KNNResult<TVector, TDistance>>[] BatchRangeQuery(IList<TVector> queries, TDistance range, Func<TVector, bool>? filterFnc = null, int layer = 0)
        {
            var result = new List<KNNResult<TVector, TDistance>>[queries.Count];
            Parallel.For(0, queries.Count, (i) =>
            {
                result[i] = RangeQuery(queries[i], range, filterFnc, layer);
            });
            return result;
        }

        /// <summary>
        /// Perform knn query over all layers in graph. Optionally provide range of layers with max and min layer parameters.
        /// </summary>
        public List<KNNResult<TVector, TDistance>>[] MultiLayerKnnQuery(TVector query, int k, int maxLayer = int.MaxValue, int minLayer = 0)
        {
            // TODO: Add checks for invalid max and min layer
            if (data.Count <= 0 || k < 1) return [];

            var ep = data.EntryPoint.MaxLayer >= maxLayer ? navigator.FindEntryPointQuery(maxLayer, query) : data.EntryPoint;
            var result = new List<KNNResult<TVector, TDistance>>[Math.Min(ep.MaxLayer, maxLayer) + 1];
            for (int layer = Math.Min(ep.MaxLayer, maxLayer); layer >= minLayer; layer--)
            {
                var candidates = navigator.SearchLayerQuery(ep.Id, layer, k, query).OrderBy(c => c.Dist).ToList();
                ep = data.Nodes[candidates[0].Id];
                result[layer] = candidates.Count > 1 ? candidates[1..].ConvertAll(CandidateToResult) : new();
            }
            return result;
        }

        /// <summary>
        /// Get statistical information about graph structure
        /// </summary>
        public HNSWInfo GetInfo()
        {
            var activeNodes = data.ActiveIds.Select(id => data.Nodes[id]).ToArray();
            return new HNSWInfo(activeNodes, data.GetTopLayer(), parameters.AllowRemovals);
        }

        /// <summary>
        /// Get the number of weakly connected components at each layer of the graph.
        /// The returned array is indexed by layer id, starting at layer zero.
        /// </summary>
        public int[] GetConnectedComponentCounts()
        {
            return navigator.GetConnectedComponentCounts();
        }

        /// <summary>
        /// Serialize the graph snapshot image to a file.
        /// </summary>
        public void Serialize(string filePath)
        {
            using (var file = File.Create(filePath))
            {
                var snapshot = new HNSWIndexSnapshot<TVector, TDistance>(parameters, data);
                Serializer.Serialize(file, snapshot);
            }
        }

        /// <summary>
        /// Reconstruct the graph from a serialized snapshot image.
        /// </summary>
        public static HNSWIndex<TVector, TDistance> Deserialize(Func<TVector, TVector, TDistance> distFnc, string filePath)
        {
            using (var file = File.OpenRead(filePath))
            {
                var snapshot = Serializer.Deserialize<HNSWIndexSnapshot<TVector, TDistance>>(file);
                return new HNSWIndex<TVector, TDistance>(distFnc, snapshot);
            }
        }

        /// <summary>
        /// Get list of items inserted into the graph structure
        /// </summary>
        public List<TVector> Items()
        {
            return data.ActiveIds.Select(id => data.Items[id]).ToList();
        }

        /// <summary>
        /// Get list of ids of items inserted into the graph structure
        /// </summary>
        public List<int> Ids()
        {
            return data.ActiveIds.ToList();
        }

        /// <summary>
        /// Get the number of items in the graph structure.
        /// </summary>
        public int Count => data.Count;

        private KNNResult<TVector, TDistance> CandidateToResult(NodeDistance<TDistance> nodeDistance)
        {
            return new KNNResult<TVector, TDistance>(nodeDistance.Id, data.Items[nodeDistance.Id], nodeDistance.Dist);
        }

        private void OnDataResized(object? sender, ReallocateEventArgs e)
        {
            navigator.OnReallocate(e.NewCapacity);
        }
    }
}
