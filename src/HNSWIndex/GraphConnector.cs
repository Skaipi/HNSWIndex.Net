using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

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
        /// Remove node from the graph at all layers.
        /// After this operation no other node in graph will point to provided item.
        /// </summary>
        internal void RemoveNodeConnections(Node item)
        {
            for (int layer = item.MaxLayer; layer >= 0; layer--)
            {
                using (data.GraphLocker.LockNodeNeighbourhood(item, layer))
                {
                    // Handle EP removal
                    if (item.Id == data.EntryPointId)
                    {
                        var replacementFound = data.TryReplaceEntryPoint(layer);
                        if (!replacementFound && layer == 0) data.EntryPointId = -1;
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

                    localCandidates.Add(new NodeDistance<TDistance>(candidateId, data.Distance(candidateId, activeNodeId)));
                }

                //TODO: Maybe use heuristic fuction here
                localCandidates.Sort(Heuristic<TDistance>.CloserFirst);
                for (int j = 0; j < localCandidates.Count && activeNeighbours.Count < data.MaxEdges(layer); j++)
                {
                    var candidate = localCandidates[j];
                    var candidateId = candidate.Id;
                    var candidateDist = candidate.Dist;

                    bool acceptable = true;
                    for (int n = 0; n < activeNeighbours.Count; n++)
                    {
                        var neighborId = activeNeighbours[n];
                        if (data.Distance(neighborId, candidateId) < candidateDist) { acceptable = false; break; }
                    }

                    if (acceptable)
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
            var bestNeighboursIds = parameters.Heuristic(topCandidates, data.Distance, data.MaxEdges(layer));
            // lock is already acquired
            currNode.OutEdges[layer] = bestNeighboursIds;
            if (parameters.AllowRemovals) currNode.InEdges[layer].AddRange(bestNeighboursIds);

            for (int i = 0; i < bestNeighboursIds.Count; ++i)
            {
                int newNeighbourId = bestNeighboursIds[i];
                var neighbor = data.Nodes[newNeighbourId];
                lock (neighbor.OutEdgesLock)
                {
                    if (parameters.AllowRemovals) neighbor.InEdges[layer].Add(currNode.Id);
                    neighbor.OutEdges[layer].Add(currNode.Id);

                    if (neighbor.OutEdges[layer].Count > data.MaxEdges(layer))
                    {
                        PruneOverflow(neighbor, layer);
                    }
                }
            }

            return bestNeighboursIds[0];
        }

        /// <summary>
        /// Prune overflow of neighbors using heuristic function.
        /// </summary>
        private void PruneOverflow(Node node, int layer)
        {
            int removedCount = 0;
            int addedCount = 0;
            List<int> oldOut;
            List<int> newOut;

            oldOut = node.OutEdges[layer];
            var candidates = oldOut;
            var candidatesDistances = new NodeDistance<TDistance>[candidates.Count];
            for (int i = 0; i < candidates.Count; i++)
            {
                int cand = candidates[i];
                candidatesDistances[i] = new NodeDistance<TDistance>(cand, data.Distance(cand, node.Id));
            }
            newOut = parameters.Heuristic(candidatesDistances, data.Distance, data.MaxEdges(layer));

            int commonLen = oldOut.Count;
            Span<int> removed = commonLen <= 128 ? stackalloc int[commonLen] : new int[commonLen];
            Span<int> added = commonLen <= 128 ? stackalloc int[commonLen] : new int[commonLen];

            for (int i = 0; i < oldOut.Count; i++)
            {
                int id = oldOut[i];
                bool keep = false;
                for (int j = 0; j < newOut.Count; j++) { if (newOut[j] == id) { keep = true; break; } }
                if (!keep) removed[removedCount++] = id;
            }

            for (int i = 0; i < newOut.Count; i++)
            {
                int id = newOut[i];
                bool existed = false;
                for (int j = 0; j < oldOut.Count; j++) { if (oldOut[j] == id) { existed = true; break; } }
                if (!existed) added[addedCount++] = id;
            }

            node.OutEdges[layer] = newOut;

            // NOTE: reverse edges and, hence, InLocks are needed only if removals happen. They may impose serious parrallelization bottleneck.
            if (parameters.AllowRemovals == false) return;

            for (int i = 0; i < removedCount; i++)
            {
                int rid = removed[i];
                var nb = data.Nodes[rid];
                lock (nb.InEdgesLock)
                {
                    var inList = nb.InEdges[layer];
                    int idx = IndexOf(inList, node.Id);
                    if (idx >= 0)
                    {
                        int last = inList.Count - 1;
                        inList[idx] = inList[last];
                        inList.RemoveAt(last);
                    }
                }
            }

            for (int i = 0; i < addedCount; i++)
            {
                int aid = added[i];
                var nb = data.Nodes[aid];
                lock (nb.InEdgesLock)
                {
                    nb.InEdges[layer].Add(node.Id);
                }
            }
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int IndexOf(List<int> list, int value)
        {
            var span = CollectionsMarshal.AsSpan(list);
            for (int i = 0; i < span.Length; i++) if (span[i] == value) return i;
            return -1;
        }

        /// <summary>
        /// Forget node as neighbor by incomming edge.
        /// </summary>
        private void WipeRelationsWithNode(Node node, int layer)
        {
            // This is done in removal context. Locks are already acquired
            foreach (var neighbourId in node.OutEdges[layer])
            {
                data.Nodes[neighbourId].InEdges[layer].Remove(node.Id);
            }
        }
    }
}
