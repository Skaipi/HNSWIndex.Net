﻿using System.Numerics;

namespace HNSWIndex
{
    public class HNSWIndex<TLabel, TDistance> where TDistance : struct, IFloatingPoint<TDistance>
    {
        private Func<TLabel, TLabel, TDistance> distanceFnc;

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
            distanceFnc = distFnc;
            parameters = hnswParameters;

            data = new GraphData<TLabel, TDistance>(distFnc, hnswParameters);
            navigator = new GraphNavigator<TLabel, TDistance>(data);
            connector = new GraphConnector<TLabel, TDistance>(data, navigator, hnswParameters);

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

            lock (data.Nodes[itemId].OutEdgesLock)
            {
                connector.ConnectNewNode(itemId);
            }
            return itemId;
        }

        /// <summary>
        /// Remove item with given index from graph structure
        /// NOTE: This method does not support parallel removals
        /// </summary>
        public void Remove(int itemIndex)
        {
            if (itemIndex == data.EntryPointId)
            {
                data.RemoveEntryPoint();
            }

            data.RemoveItem(itemIndex);
            connector.RemoveConnections(itemIndex);
        }

        /// <summary>
        /// Get list of items inserted into the graph structure
        /// </summary>
        public List<TLabel> Items()
        {
            return data.Items.Values.ToList();
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
        // Optionally probide filter function to ignore certain labels.
        /// </summary>
        public List<KNNResult<TLabel, TDistance>> KnnQuery(TLabel query, int k, Func<TLabel, bool>? filterFnc = null)
        {
            if (data.Nodes.Count - data.RemovedIndexes.Count <= 0) return new List<KNNResult<TLabel, TDistance>>();

            Func<int, bool> indexFilter = _ => true;
            if (filterFnc is not null)
                indexFilter = (index) => filterFnc(data.Items[index]);


            TDistance queryDistance(int nodeId, TLabel label)
            {
                return distanceFnc(data.Items[nodeId], label);
            }

            var neighboursAmount = Math.Max(parameters.MinNN, k);
            var distCalculator = new DistanceCalculator<TLabel, TDistance>(queryDistance, query);
            var ep = navigator.FindEntryPoint(0, distCalculator);
            var topCandidates = navigator.SearchLayer(ep.Id, 0, neighboursAmount, distCalculator, indexFilter);

            if (k < neighboursAmount)
            {
                return topCandidates.OrderBy(c => c.Dist).Take(k).ToList().ConvertAll(c => new KNNResult<TLabel, TDistance>(c.Id, data.Items[c.Id], c.Dist));
            }
            return topCandidates.ConvertAll(c => new KNNResult<TLabel, TDistance>(c.Id, data.Items[c.Id], c.Dist));
        }

        /// <summary>
        /// Get statistical information about graph structure
        /// </summary>
        public HNSWInfo GetInfo()
        {
            return new HNSWInfo(data.Nodes, data.RemovedIndexes, data.GetTopLayer());
        }

        private void OnDataResized(object? sender, ReallocateEventArgs e)
        {
            navigator.OnReallocate(e.NewCapacity);
        }
    }
}
