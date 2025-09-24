using System.Collections;
using System.Numerics;
using ProtoBuf;

namespace HNSWIndex
{
    public class HNSWIndex<TLabel, TDistance> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance> where TLabel : IList
    {
        private readonly HNSWParameters<TDistance> parameters;

        private readonly GraphData<TLabel, TDistance> data;

        private readonly GraphConnector<TLabel, TDistance> connector;

        private readonly GraphNavigator<TLabel, TDistance> navigator;

        /// <summary>
        /// Construct KNN search graph with arbitrary distance function
        /// </summary>
        public HNSWIndex(Func<TLabel, TLabel, TDistance> distFnc, HNSWParameters<TDistance>? hnswParameters = null)
        {
            hnswParameters ??= new HNSWParameters<TDistance>();
            parameters = hnswParameters;

            data = new GraphData<TLabel, TDistance>(distFnc, hnswParameters);
            navigator = new GraphNavigator<TLabel, TDistance>(data);
            connector = new GraphConnector<TLabel, TDistance>(data, navigator, hnswParameters);

            data.Reallocated += OnDataResized;
        }

        /// <summary>
        /// Construct KNN search graph from serialized snapshot.
        /// </summary>
        internal HNSWIndex(Func<TLabel, TLabel, TDistance> distFnc, HNSWIndexSnapshot<TLabel, TDistance> snapshot)
        {
            if (snapshot.Parameters is null)
                throw new ArgumentNullException(nameof(snapshot.Parameters), "Parameters cannot be null during deserialization.");

            if (snapshot.DataSnapshot is null)
                throw new ArgumentNullException(nameof(snapshot.DataSnapshot), "Data cannot be null during deserialization.");

            parameters = snapshot.Parameters;
            data = new GraphData<TLabel, TDistance>(snapshot.DataSnapshot, distFnc, snapshot.Parameters);

            navigator = new GraphNavigator<TLabel, TDistance>(data);
            connector = new GraphConnector<TLabel, TDistance>(data, navigator, parameters);

            data.Reallocated += OnDataResized;
        }

        /// <summary>
        /// Add new item with given label to the graph.
        /// </summary>
        public int Add(TLabel item)
        {
            var itemId = -1;
            lock (data.indexLock)
            {
                itemId = data.AddItem(item);
            }
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
        public int[] Add(List<TLabel> items)
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
        /// Update items at given indexes with new labels 
        /// </summary>
        public void Update(IList<int> indexes, IList<TLabel> labels)
        {
            connector.UpdateOldConnections(indexes, labels);
        }

        /// <summary>
        /// Get list of items inserted into the graph structure
        /// </summary>
        public List<TLabel> Items()
        {
            return data.Items.ToList();
        }

        /// <summary>
        /// Directly access graph structure at given layer
        /// </summary>
        public GraphLayer GetGraphLayer(int layer)
        {
            return new GraphLayer(data.Nodes, layer);
        }

        /// <summary>
        /// Get K nearest neighbours of query point. 
        /// Optionally provide filter function to ignore certain labels.
        /// Layer parameters indicates at which layer search should be performed (0 - base layer)
        /// </summary>
        public List<KNNResult<TLabel, TDistance>> KnnQuery(TLabel query, int k, Func<TLabel, bool>? filterFnc = null, int layer = 0)
        {
            if (data.Count <= 0 || k < 1) return new List<KNNResult<TLabel, TDistance>>();

            Func<int, bool> indexFilter = _ => true;
            if (filterFnc is not null)
                indexFilter = (index) => filterFnc(data.Items[index]);

            var neighboursAmount = Math.Max(parameters.MinNN, k);
            var ep = navigator.FindEntryPoint(layer, query, false);
            var topCandidates = navigator.SearchLayer(ep.Id, layer, neighboursAmount, query, indexFilter, false);

            if (k < neighboursAmount)
            {
                return topCandidates.OrderBy(c => c.Dist).Take(k).ToList().ConvertAll(CandidateToResult);
            }
            return topCandidates.ConvertAll(CandidateToResult);
        }

        /// <summary>
        /// Get all neighbours of query point which are within range distance.
        /// Optionally provide filter function to ignore certain labels.
        /// Layer parameters indicates at which layer search should be performed (0 - base layer)
        /// </summary>
        public List<KNNResult<TLabel, TDistance>> RangeQuery(TLabel query, TDistance range, Func<TLabel, bool>? filterFnc = null, int layer = 0)
        {
            if (data.Count <= 0) return new List<KNNResult<TLabel, TDistance>>();

            Func<int, bool> indexFilter = _ => true;
            if (filterFnc is not null)
                indexFilter = (index) => filterFnc(data.Items[index]);

            var ep = navigator.FindEntryPoint(layer, query, false);
            var topCandidates = navigator.SearchLayerRange(ep.Id, layer, range, query, indexFilter, false);
            return topCandidates.ConvertAll(CandidateToResult);
        }

        /// <summary>
        /// Perform knn query over all layers in graph. Optionally provide range of layers with max and min layer parameters.
        /// </summary>
        public List<KNNResult<TLabel, TDistance>>[] MultiLayerKnnQuery(TLabel query, int k, int maxLayer = int.MaxValue, int minLayer = 0)
        {
            // TODO: Add checks for invalid max and min layer
            if (data.Count <= 0 || k < 1) return [];

            var ep = data.EntryPoint.MaxLayer >= maxLayer ? navigator.FindEntryPoint(maxLayer, query, false) : data.EntryPoint;
            var result = new List<KNNResult<TLabel, TDistance>>[Math.Min(ep.MaxLayer, maxLayer) + 1];
            for (int layer = Math.Min(ep.MaxLayer, maxLayer); layer >= minLayer; layer--)
            {
                var candidates = navigator.SearchLayer(ep.Id, layer, k, query, null, false);
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
            return new HNSWInfo(data.Nodes, data.RemovedIndexes, data.GetTopLayer());
        }

        /// <summary>
        /// Serialize the graph snapshot image to a file.
        /// </summary>
        public void Serialize(string filePath)
        {
            using (var file = File.Create(filePath))
            {
                var snapshot = new HNSWIndexSnapshot<TLabel, TDistance>(parameters, data);
                Serializer.Serialize(file, snapshot);
            }
        }

        /// <summary>
        /// Reconstruct the graph from a serialized snapshot image.
        /// </summary>
        public static HNSWIndex<TLabel, TDistance> Deserialize(Func<TLabel, TLabel, TDistance> distFnc, string filePath)
        {
            using (var file = File.OpenRead(filePath))
            {
                var snapshot = Serializer.Deserialize<HNSWIndexSnapshot<TLabel, TDistance>>(file);
                return new HNSWIndex<TLabel, TDistance>(distFnc, snapshot);
            }
        }

        /// <summary>
        /// Stateful setter for CollectionSize parameter.
        /// </summary>
        public void SetCollectionSize(int collectionSize) => parameters.CollectionSize = collectionSize;

        /// <summary>
        /// Stateful setter for MaxEdges parameter sometimes mentioned as `M`.
        /// </summary>
        public void SetMaxEdges(int maxEdges) => parameters.MaxEdges = maxEdges;

        /// <summary>
        /// Stateful setter for MaxCandidates parameter sometimes mentioned as `ef_search`.
        /// </summary>
        public void SetMaxCandidates(int MaxCandidates) => parameters.MaxCandidates = MaxCandidates;

        /// <summary>
        /// Stateful setter for DistributionRate parameter.
        /// </summary>
        public void SetDistributionRate(float distRate) => parameters.DistributionRate = distRate;

        /// <summary>
        /// Stateful setter for RandomSeed parameter.
        /// </summary>
        public void SetRandomSeed(int randomSeed) => parameters.RandomSeed = randomSeed;

        /// <summary>
        /// Stateful setter for MinNN parameter sometimes mentioned as `ef`.
        /// </summary>
        public void SetMinNN(int minNN) => parameters.MinNN = minNN;

        /// <summary>
        /// Stateful setter for ZeroLayerGuaranteed parameter.
        /// </summary>
        public void SetZeroLayerGuaranteed(bool zeroLayer) => parameters.ZeroLayerGuaranteed = zeroLayer;

        private KNNResult<TLabel, TDistance> CandidateToResult(NodeDistance<TDistance> nodeDistance)
        {
            return new KNNResult<TLabel, TDistance>(nodeDistance.Id, data.Items[nodeDistance.Id], nodeDistance.Dist);
        }

        private void OnDataResized(object? sender, ReallocateEventArgs e)
        {
            navigator.OnReallocate(e.NewCapacity);
        }
    }
}
