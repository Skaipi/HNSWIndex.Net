using System.Collections;
using System.Collections.Concurrent;
using System.Numerics;

namespace HNSWIndex
{
    internal class GraphConnector<TLabel, TDistance> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance>
    {
        private static Func<int, bool> noFilter = _ => true;
        private GraphData<TLabel, TDistance> data;
        private GraphNavigator<TLabel, TDistance> navigator;
        private HNSWParameters<TDistance> parameters;

        internal GraphConnector(GraphData<TLabel, TDistance> graphData, GraphNavigator<TLabel, TDistance> graphNavigator, HNSWParameters<TDistance> hnswParams)
        {
            data = graphData;
            navigator = graphNavigator;
            parameters = hnswParams;
        }

        /// <summary>
        /// Establish connections from newly inserted node to the graph.
        /// If graph is empty, create new graph from a node.
        /// </summary>
        internal void ConnectNewNode(int nodeId)
        {
            // If this is new ep we keep lock for entire Add Operation
            Monitor.Enter(data.entryPointLock);
            if (data.EntryPointId < 0)
            {
                data.EntryPointId = nodeId;
                Monitor.Exit(data.entryPointLock);
                return;
            }

            var currNode = data.Nodes[nodeId];
            if (currNode.MaxLayer > data.GetTopLayer())
            {
                AddNewConnections(currNode);
                data.EntryPointId = nodeId;
                Monitor.Exit(data.entryPointLock);
            }
            else
            {
                Monitor.Exit(data.entryPointLock);
                AddNewConnections(currNode);
            }
        }

        /// <summary>
        /// Perform update of connections on provided list of indexes 
        /// </summary>
        internal void UpdateOldConnections(IList<int> indexes, IList<TLabel> labels)
        {
            if (indexes.Count != labels.Count) throw new ArgumentException("Update collections size mismatch");

            var dirtyByIndex = DisconnectDirtyEdges(indexes, labels);
            ReconnectDirtyNodes(dirtyByIndex);
        }

        /// <summary>
        /// Remove node from the graph at all layers.
        /// After this operation no other node in graph will point to provided item.
        /// </summary>
        internal void RemoveNodeConnections(Node item)
        {
            // TODO: Think about different method for handling side effects which have to be done under neighborhood lock.
            for (int layer = item.MaxLayer; layer >= 0; layer--)
            {
                using (data.GraphLocker.LockNodeNeighbourhood(item, layer))
                {
                    // Handle EP removal
                    if (item.Id == data.EntryPointId)
                    {
                        var replacementFound = data.TryReplaceEntryPoint(layer);
                        if (!replacementFound && layer == 0)
                        {
                            // Take current removal into account
                            data.EntryPointId = -1;
                        }
                    }
                    RemoveConnectionsAtLayer(item, layer);
                    if (layer == 0) data.RemoveItem(item.Id); // Remove label before leaving locks
                }
            }
        }

        /// <summary>
        /// Remove node from the graph at given layer.
        /// </summary>
        private void RemoveConnectionsAtLayer(Node removedNode, int layer)
        {
            WipeRelationsWithNode(removedNode, layer);

            var candidates = removedNode.OutEdges[layer];
            for (int i = 0; i < removedNode.InEdges[layer].Count; i++)
            {
                var activeNodeId = removedNode.InEdges[layer][i];
                var activeNode = data.Nodes[activeNodeId];
                var activeNeighbours = activeNode.OutEdges[layer];
                RemoveOutEdge(activeNode, removedNode, layer);

                // Select candidates for active node
                var localCandidates = new List<NodeDistance<TDistance>>();
                for (int j = 0; j < candidates.Count; j++)
                {
                    var candidateId = candidates[j];
                    if (candidateId == activeNodeId || activeNeighbours.Contains(candidateId))
                        continue;

                    localCandidates.Add(new NodeDistance<TDistance> { Id = candidateId, Dist = data.Distance(candidateId, activeNodeId) });
                }

                var candidatesHeap = new BinaryHeap<NodeDistance<TDistance>>(localCandidates, Heuristic<TDistance>.CloserFirst);
                while (candidatesHeap.Count > 0 && activeNeighbours.Count < data.MaxEdges(layer))
                {
                    var candidate = candidatesHeap.Pop();
                    if (activeNeighbours.TrueForAll((n) => data.Distance(candidate.Id, n) > candidate.Dist))
                    {
                        activeNode.OutEdges[layer].Add(candidate.Id);
                        data.Nodes[candidate.Id].InEdges[layer].Add(activeNodeId);
                    }
                }
            }
        }

        /// <summary>
        /// Remove outgoing edge from node to invalid target.
        /// This operation is assumed to work under neighborhood lock. 
        /// </summary>
        private void RemoveOutEdge(Node target, Node badNeighbour, int layer)
        {
            target.OutEdges[layer].Remove(badNeighbour.Id);
        }


        /// <summary>
        /// Mark edges with outdated neighborhood as dirty and disconnect them from the graph.
        /// Return dictionary with ids of dirty nodes valued by ditry level.
        /// </summary>
        private ConcurrentDictionary<int, int> DisconnectDirtyEdges(IList<int> indexes, IList<TLabel> labels)
        {
            // index -> highest layer that needs reinsertion, -1 means clean
            var dirtyByIndex = new ConcurrentDictionary<int, int>(Environment.ProcessorCount * 2, indexes.Count);
            var cleanAnchors = new int[data.GetTopLayer() + 1];
            Array.Fill(cleanAnchors, data.EntryPointId);

            // Decide which layers to rewire and temporarily disconnect at those layers
            Parallel.For(0, indexes.Count, (i) =>
            {
                var index = indexes[i];
                var newLabel = labels[i];
                var oldLabel = data.Items[index];
                var difference = data.Distance(newLabel, oldLabel);
                var node = data.Nodes[index];

                if (TDistance.IsZero(difference)) return;

                for (int layer = 0; layer <= node.MaxLayer; layer++)
                {
                    // Update on significant difference 
                    using (data.GraphLocker.LockNodeNeighbourhood(node, layer))
                    {
                        var outs = node.OutEdges[layer];
                        var isDirtyAlready = dirtyByIndex.ContainsKey(index);
                        if (outs.Count == 0) return; // Most likely single point at layer

                        // TODO: Replace linq call with manual min finding
                        var minEdge = outs.Count > 0 ? outs.Min(neighbor => data.Distance(index, neighbor)) : TDistance.MaxValue;
                        if (difference < minEdge) return; // Skip layer w/o significant change

                        dirtyByIndex[index] = layer;

                        // Move ep if change would invalid its status
                        // TODO: Replace linq call with manual maxby finding
                        if (index == cleanAnchors[layer])
                        {
                            cleanAnchors[layer] = outs.Count > 0 ? outs.MaxBy(id => data.Nodes[id].OutEdges[layer].Count) : -1;
                        }
                        RemoveConnectionsAtLayer(node, layer);
                        ResetNodeConnectionsAtLayer(node, layer);
                    }
                }
                // swap the label
                data.Items[index] = newLabel;
            });

            // Resotore good ep
            if (dirtyByIndex.GetValueOrDefault(data.EntryPointId, -1) >= 0)
            {
                for (int layer = dirtyByIndex[data.EntryPointId]; layer >= 0; layer--)
                {
                    var restorationEntry = data.Nodes[cleanAnchors[layer]];
                    ConnectAtLayer(data.EntryPoint, restorationEntry, layer);
                }
                dirtyByIndex[data.EntryPointId] = -1;
            }

            return dirtyByIndex;
        }

        /// <summary>
        /// Reestablish connections from dirty nodes.
        /// </summary>
        private void ReconnectDirtyNodes(ConcurrentDictionary<int, int> dirtyByIndex)
        {
            Parallel.ForEach(dirtyByIndex, (kvp) =>
            {
                var index = kvp.Key;
                var topLayer = kvp.Value;
                var node = data.Nodes[index];
                if (topLayer < 0) return;

                // Perform reconnect
                lock (node.OutEdgesLock)
                {
                    // Select best peer which has connections at rebuild layer
                    Func<int, int, bool> EntryFilter = (cand, layer) => cand != index && dirtyByIndex.GetValueOrDefault(cand, -1) < layer;
                    var bestPeer = navigator.FindEntryPoint(topLayer, data.Items[index], true, cand => EntryFilter(cand, topLayer));
                    for (int layer = topLayer; layer >= 0; layer--)
                    {
                        if (!EntryFilter(bestPeer.Id, layer)) throw new InvalidOperationException("Dirty entry point while reconnecting");
                        if (layer > 0)
                        {
                            var nextLayer = layer - 1;
                            var nextEntryCandidateId = ConnectAtLayer(node, bestPeer, layer, cand => EntryFilter(cand, nextLayer));
                            bestPeer = nextEntryCandidateId >= 0 ? data.Nodes[nextEntryCandidateId] : data.EntryPoint;
                        }
                        else ConnectAtLayer(node, bestPeer, layer);

                        // update status
                        dirtyByIndex.TryGetValue(index, out var a);
                        dirtyByIndex.TryUpdate(index, a - 1, a);
                    }
                    dirtyByIndex.TryUpdate(index, -1, 0);
                }
            });
        }

        /// <summary>
        /// Establish connections to node.
        /// </summary>
        internal void AddNewConnections(Node currNode)
        {
            var bestPeer = navigator.FindEntryPoint(currNode.MaxLayer, data.Items[currNode.Id]);

            for (int layer = Math.Min(currNode.MaxLayer, data.GetTopLayer()); layer >= 0; --layer)
            {
                int nextClosestEntryPointId = ConnectAtLayer(currNode, bestPeer, layer);
                bestPeer = data.Nodes[nextClosestEntryPointId];
            }
        }

        /// <summary>
        /// Establish connections to node at given layer and return best peer.
        /// Optionally, provide filter function to discriminate certain solutions from ep status.
        /// </summary>
        internal int ConnectAtLayer(Node currNode, Node bestPeer, int layer, Func<int, bool>? filterFnc = null)
        {
            filterFnc ??= noFilter;

            var topCandidates = navigator.SearchLayer(bestPeer.Id, layer, parameters.MaxCandidates, data.Items[currNode.Id]);
            var bestNeighboursIds = parameters.Heuristic(topCandidates, data.Distance, data.MaxEdges(layer));

            for (int i = 0; i < bestNeighboursIds.Count; ++i)
            {
                int newNeighbourId = bestNeighboursIds[i];
                Connect(currNode, data.Nodes[newNeighbourId], layer);
                Connect(data.Nodes[newNeighbourId], currNode, layer);
            }

            return bestNeighboursIds.Where(filterFnc).FirstOrDefault(-1);
        }

        /// <summary>
        /// Connect two nodes and handle overflow of edges.
        /// </summary>
        private void Connect(Node node, Node neighbour, int layer)
        {
            lock (node.OutEdgesLock)
            {
                // Try simple addition
                node.OutEdges[layer].Add(neighbour.Id);
                lock (neighbour.InEdgesLock)
                {
                    neighbour.InEdges[layer].Add(node.Id);
                }
                // Connections exceeded limit from parameters
                if (node.OutEdges[layer].Count > data.MaxEdges(layer))
                {
                    WipeRelationsWithNode(node, layer);
                    RecomputeConnections(node, node.OutEdges[layer], layer);
                    SetRelationsWithNode(node, layer);
                }
            }
        }

        /// <summary>
        /// Handle overflow of neighbors using heuristic function.
        /// </summary>
        private void RecomputeConnections(Node node, List<int> candidates, int layer)
        {
            var candidatesDistances = new List<NodeDistance<TDistance>>(candidates.Count);
            foreach (var neighbourId in candidates)
                candidatesDistances.Add(new NodeDistance<TDistance> { Dist = data.Distance(neighbourId, node.Id), Id = neighbourId });
            var newNeighbours = parameters.Heuristic(candidatesDistances, data.Distance, data.MaxEdges(layer));
            node.OutEdges[layer] = newNeighbours;
        }

        /// <summary>
        /// Clear nodes adjacency list across all layers.
        /// </summary>
        internal void ResetNodeConnections(Node node)
        {
            for (int layer = node.MaxLayer; layer >= 0; layer--)
            {
                ResetNodeConnectionsAtLayer(node, layer);
            }
        }

        /// <summary>
        /// Clear node adjacencty list at specified layer.
        /// </summary>
        internal void ResetNodeConnectionsAtLayer(Node node, int layer)
        {
            node.OutEdges[layer] = new List<int>(data.MaxEdges(layer) + 1);
            node.InEdges[layer] = new List<int>(data.MaxEdges(layer) + 1);
        }

        /// <summary>
        /// Forget node as neighbor by incomming edge.
        /// </summary>
        private void WipeRelationsWithNode(Node node, int layer)
        {
            lock (node.OutEdgesLock)
            {
                foreach (var neighbourId in node.OutEdges[layer])
                {
                    lock (data.Nodes[neighbourId].InEdgesLock)
                    {
                        data.Nodes[neighbourId].InEdges[layer].Remove(node.Id);
                    }
                }
            }
        }

        /// <summary>
        /// Set incoming edges to node based on its adjacency list.
        /// </summary>
        private void SetRelationsWithNode(Node node, int layer)
        {
            lock (node.OutEdgesLock)
            {
                foreach (var neighbourId in node.OutEdges[layer])
                {
                    lock (data.Nodes[neighbourId].InEdgesLock)
                    {
                        data.Nodes[neighbourId].InEdges[layer].Add(node.Id);
                    }
                }
            }
        }
    }
}
