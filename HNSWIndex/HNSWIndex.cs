using System.Collections;
using System.Collections.Concurrent;
using System.Numerics;
using ProtoBuf;

namespace HNSWIndex
{
    public class HNSWIndex<TLabel, TDistance> where TDistance : struct, IFloatingPoint<TDistance> where TLabel : IList
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
            if (indexes.Count != labels.Count) throw new ArgumentException("Update collections size mismatch");

            var originalEpId = data.EntryPointId;
            var maxLayerToFix = new int[indexes.Count]; // Top layer where update occured
            Array.Fill(maxLayerToFix, -1);

            // Decide which layers to rewire and temporarily disconnect at those layers
            Parallel.For(0, indexes.Count, (i) =>
            {
                var index = indexes[i];
                var newLabel = labels[i];
                var oldLabel = data.Items[index];
                var difference = data.Distance(newLabel, oldLabel);
                var node = data.Nodes[index];

                var firstNonEmptyLayer = -1;
                for (int layer = node.MaxLayer; layer >= 0; layer--)
                {
                    // Update on significant difference 
                    using (data.GraphLocker.LockNodeNeighbourhood(node, layer))
                    {
                        var outs = node.OutEdges[layer];
                        if (outs.Count == 0) continue; // Most likely single point at layer
                        if (firstNonEmptyLayer == -1) firstNonEmptyLayer = layer;

                        var minEdge = outs.Min(neighbor => data.Distance(index, neighbor));
                        if (difference < minEdge) continue; // Skip layer w/o significant change
                        if (maxLayerToFix[i] == -1) maxLayerToFix[i] = layer;

                        // Move ep if change would invalid its status
                        if (index == data.EntryPointId && maxLayerToFix[i] == firstNonEmptyLayer)
                        {
                            var replacementFound = data.TryReplaceEntryPoint(layer);
                            if (!replacementFound)
                            {
                                data.EntryPointId = -1;
                            }
                        }
                        connector.RemoveConnectionsAtLayer(node, layer);
                        connector.ResetNodeConnectionsAtLayer(node, layer);
                    }
                }
                // swap the label
                data.Items[index] = newLabel;
            });

            var updateMap = new ConcurrentDictionary<int, int>(65536, indexes.Count);
            for (int i = 0; i < indexes.Count; i++)
                if (maxLayerToFix[i] >= 0) updateMap[indexes[i]] = maxLayerToFix[i];


            // Resotore good ep
            if (data.EntryPointId < 0)
            {
                data.EntryPointId = originalEpId;
            }
            if (updateMap.GetValueOrDefault(originalEpId, -1) >= 0)
            {
                // Fix if layer is not empty
                var originalEp = data.Nodes[originalEpId];
                var epMaxLayerToFix = updateMap[originalEpId];
                Func<int, bool> epLayerFilter = cand => cand != originalEpId && updateMap.GetValueOrDefault(cand, -1) < epMaxLayerToFix;

                var restorationEntry = navigator.FindEntryPoint(updateMap[originalEpId], data.Items[originalEpId], true, epLayerFilter);
                for (int layer = Math.Min(updateMap[originalEpId], data.GetTopLayer()); layer >= 0; layer--)
                {
                    if (!epLayerFilter(restorationEntry.Id)) break;
                    var nextEntryCandidate = -1;
                    if (layer > 0)
                    {
                        epLayerFilter = cand => cand != originalEpId && updateMap.GetValueOrDefault(cand, -1) < layer - 1;
                        nextEntryCandidate = connector.ConnectAtLayer(originalEp, restorationEntry, layer, epLayerFilter);
                    }
                    else
                    {
                        epLayerFilter = cand => cand != originalEpId && updateMap.GetValueOrDefault(cand, -1) < 0;
                        nextEntryCandidate = connector.ConnectAtLayer(originalEp, restorationEntry, layer, epLayerFilter);
                    }

                    restorationEntry = data.Nodes[nextEntryCandidate];
                }
                data.EntryPointId = originalEpId;
                updateMap[originalEpId] = -1;
            }

            // Insert node with new label with the same index. At this point we have fully restored ep that can be used to navigate graph.
            Parallel.For(0, indexes.Count, (i) =>
            {
                var updateLayer = maxLayerToFix[i];
                if (updateLayer < 0) return;

                var index = indexes[i];
                var node = data.Nodes[index];
                if (index == data.EntryPointId) return;

                // Perform reconnect
                lock (node.OutEdgesLock)
                {
                    Func<int, bool> entryFilter = cand => cand != index && updateMap.GetValueOrDefault(cand, -1) < updateLayer;
                    var bestPeer = navigator.FindEntryPoint(updateLayer, data.Items[index], true, entryFilter);
                    for (int layer = updateLayer; layer >= 0; layer--)
                    {
                        if (!entryFilter(bestPeer.Id)) bestPeer = data.EntryPoint;
                        var nextEntryCandidateId = -1;
                        if (layer > 0)
                        {
                            entryFilter = cand => cand != index && updateMap.GetValueOrDefault(cand, -1) < layer - 1;
                            nextEntryCandidateId = connector.ConnectAtLayer(node, bestPeer, layer, entryFilter);
                        }
                        else
                        {
                            entryFilter = cand => cand != index && updateMap.GetValueOrDefault(cand, -1) < 0;
                            nextEntryCandidateId = connector.ConnectAtLayer(node, bestPeer, layer, entryFilter);
                        }

                        bestPeer = data.Nodes[nextEntryCandidateId];
                        updateMap[index] -= 1;
                    }
                }
            });
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
            var topCandidates = navigator.SearchLayer(ep.Id, layer, neighboursAmount, query, indexFilter, null, false);

            if (k < neighboursAmount)
            {
                return topCandidates.OrderBy(c => c.Dist).Take(k).ToList().ConvertAll(c => new KNNResult<TLabel, TDistance>(c.Id, data.Items[c.Id], c.Dist));
            }
            return topCandidates.ConvertAll(c => new KNNResult<TLabel, TDistance>(c.Id, data.Items[c.Id], c.Dist));
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
                ep = navigator.FindEntryAtLayer(layer, ep, query, false);
                var candidates = navigator.SearchLayer(ep.Id, layer, k, query, (index) => index != ep.Id, null, false); // Search closest neighbors except entry point
                result[layer] = candidates.ConvertAll(c => new KNNResult<TLabel, TDistance>(c.Id, data.Items[c.Id], c.Dist));
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

        private void OnDataResized(object? sender, ReallocateEventArgs e)
        {
            navigator.OnReallocate(e.NewCapacity);
        }
    }
}
