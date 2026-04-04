using System.Buffers;
using System.Numerics;

namespace HNSWIndex
{
    internal class GraphConnector<TVector, TDistance> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance>
    {
        private static Func<int, bool> noFilter = _ => true;
        private GraphData<TVector, TDistance> data;
        private GraphNavigator<TVector, TDistance> navigator;
        private HNSWParameters<TDistance> parameters;

        internal GraphConnector(GraphData<TVector, TDistance> graphData, GraphNavigator<TVector, TDistance> graphNavigator, HNSWParameters<TDistance> hnswParams)
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
        /// Remove a node from every layer.
        /// After this operation no other node in graph will point to provided item.
        /// </summary>
        internal void RemoveNodeConnections(Node item)
        {
            item.NodeLock.EnterWriteLock();
            item.IsRemoved = true;
            item.NodeLock.ExitWriteLock();

            for (int layer = item.MaxLayer; layer >= 0; layer--)
            {
                using var lockToken = data.GraphLocker.LockNodeNeighborhood(item, layer);

                ReplaceEntryPointIfNeeded(item, layer);
                RemoveConnectionsAtLayer(item, layer);
                if (layer == 0) data.RemoveItem(item.Id); // Remove label before leaving locks
            }
        }

        /// <summary>
        /// Moves the entry point status if the removed node was the entry point.
        /// </summary>
        private void ReplaceEntryPointIfNeeded(Node removedNode, int layer)
        {
            if (removedNode.Id != data.EntryPointId) return;
            if (data.TryReplaceEntryPoint(layer)) return;
            if (layer > 0) return;

            if (data.Count == 1)
            {
                data.EntryPointId = -1;
                return;
            }

            data.ForceReplaceEntryPoint();
        }

        /// <summary>
        /// Remove node from the graph at given layer.
        /// </summary>
        private void RemoveConnectionsAtLayer(Node removedNode, int layer)
        {
            DetachOutgoingReferences(removedNode, layer);

            var affectedNodes = removedNode.InEdges[layer].ToArray();
            var searchCandidates = navigator.SearchLayer(removedNode.Id, layer, parameters.RemoveMaxCandidates, data.Items[removedNode.Id], id => id != removedNode.Id);
            var localCandidates = ArrayPool<NodeDistance<TDistance>>.Shared.Rent(searchCandidates.Length);
            for (int i = 0; i < affectedNodes.Length; i++)
            {
                var affectedNodeId = affectedNodes[i];
                var activeNode = data.Nodes[affectedNodeId];
                RemoveOutEdge(activeNode, removedNode, layer);
                var affectedNodeNeighbors = activeNode.OutEdges[layer].AsSpan();

                var candidatesCount = 0;
                for (int j = 0; j < searchCandidates.Length; j++)
                {
                    var candidateId = searchCandidates[j].Id;
                    if (candidateId == affectedNodeId) continue;
                    localCandidates[candidatesCount++] = new NodeDistance<TDistance>(candidateId, data.Distance(candidateId, affectedNodeId));
                }

                var localCandidatesView = localCandidates.AsSpan(0, candidatesCount);
                var prunedCandidates = Heuristic<TDistance>.RelativeNeighborPruning(localCandidatesView, data.Distance, data.MaxEdges(layer)).AsSpan();
                for (int j = 0; j < prunedCandidates.Length; j++)
                {
                    var candidateId = prunedCandidates[j];
                    if (affectedNodeNeighbors.Contains(candidateId)) continue;

                    var candidateNode = data.Nodes[candidateId];
                    var locked = candidateNode.NodeLock.TryEnterReadLock(0);
                    try
                    {
                        if (!locked || candidateNode.IsRemoved) continue;

                        lock (activeNode.OutEdgesLock) activeNode.OutEdges[layer].Add(candidateId);
                        lock (candidateNode.InEdgesLock) candidateNode.InEdges[layer].Add(affectedNodeId);
                    }
                    finally
                    {
                        if (locked) candidateNode.NodeLock.ExitReadLock();
                    }
                }

                if (activeNode.OutEdges[layer].Count > data.MaxEdges(layer))
                {
                    lock (activeNode.OutEdgesLock) PruneOverflow(activeNode, layer);
                }
            }
            ArrayPool<NodeDistance<TDistance>>.Shared.Return(localCandidates);
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
        internal int ConnectAtLayer(Node currNode, Node bestPeer, int layer)
        {
            var topCandidates = navigator.SearchLayer(bestPeer.Id, layer, parameters.MaxCandidates, data.Items[currNode.Id]);
            var bestNeighborsIds = Heuristic<TDistance>.RelativeNeighborPruning(topCandidates, data.Distance, data.MaxEdges(layer));
            // lock is already acquired
            currNode.OutEdges[layer] = bestNeighborsIds;
            if (parameters.AllowRemovals) currNode.InEdges[layer] = new EdgeList(bestNeighborsIds);

            var bestNeighborsIdsSpan = bestNeighborsIds.AsSpan();
            for (int i = 0; i < bestNeighborsIds.Count; ++i)
            {
                int newNeighborId = bestNeighborsIdsSpan[i];
                var neighbor = data.Nodes[newNeighborId];
                lock (neighbor.OutEdgesLock)
                {
                    if (parameters.AllowRemovals)
                    {
                        lock (neighbor.InEdgesLock) neighbor.InEdges[layer].Add(currNode.Id);
                    }

                    neighbor.OutEdges[layer].Add(currNode.Id);

                    if (neighbor.OutEdges[layer].Count > data.MaxEdges(layer))
                    {
                        PruneOverflow(neighbor, layer);
                    }
                }
            }

            return bestNeighborsIdsSpan[0];
        }

        /// <summary>
        /// Prune overflow of neighbors using heuristic function.
        /// </summary>
        private void PruneOverflow(Node node, int layer)
        {
            int removedCount = 0;
            EdgeList newOut;
            var oldOutSpan = node.OutEdges[layer].AsSpan();

            var candidates = oldOutSpan;
            var candidatesDistances = ArrayPool<NodeDistance<TDistance>>.Shared.Rent(candidates.Length);
            for (int i = 0; i < candidates.Length; i++)
            {
                int cand = candidates[i];
                candidatesDistances[i] = new NodeDistance<TDistance>(cand, data.Distance(cand, node.Id));
            }
            newOut = Heuristic<TDistance>.RelativeNeighborPruning(candidatesDistances.AsSpan(0, candidates.Length), data.Distance, data.MaxEdges(layer));
            node.OutEdges[layer] = newOut;

            ArrayPool<NodeDistance<TDistance>>.Shared.Return(candidatesDistances);
            if (parameters.AllowRemovals == false) return;

            var newOutSpan = newOut.AsSpan();
            int commonLen = oldOutSpan.Length;
            Span<int> removed = commonLen <= 128 ? stackalloc int[commonLen] : new int[commonLen];

            for (int i = 0; i < oldOutSpan.Length; i++)
            {
                int id = oldOutSpan[i];
                bool keep = false;
                for (int j = 0; j < newOut.Count; j++) { if (newOutSpan[j] == id) { keep = true; break; } }
                if (!keep) removed[removedCount++] = id;
            }

            for (int i = 0; i < removedCount; i++)
            {
                int rid = removed[i];
                var nb = data.Nodes[rid];
                lock (nb.InEdgesLock)
                {
                    nb.InEdges[layer].Remove(node.Id);
                }
            }
        }

        /// <summary>
        /// Remove outgoing edge from node to invalid target.
        /// This operation is assumed to work under neighborhood lock. 
        /// </summary>
        private void RemoveOutEdge(Node target, Node badNeighbor, int layer)
        {
            lock (target.OutEdgesLock)
                target.OutEdges[layer].Remove(badNeighbor.Id);
        }

        /// <summary>
        /// Forget node as neighbor by incomming edge.
        /// </summary>
        private void DetachOutgoingReferences(Node node, int layer)
        {
            var edgesSpan = node.OutEdges[layer].AsSpan();
            for (int i = 0; i < edgesSpan.Length; i++)
            {
                var neighborId = edgesSpan[i];
                var neighbor = data.Nodes[neighborId];

                lock (neighbor.InEdgesLock)
                    neighbor.InEdges[layer].Remove(node.Id);
            }
        }
    }
}
