using System.Collections;
using System.Numerics;

namespace HNSWIndex
{
    internal class GraphConnector<TLabel, TDistance> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance> where TLabel : IList
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

        internal void RemoveConnectionsAtLayer(Node removedNode, int layer)
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

        internal void ResetNodeConnections(Node node)
        {
            for (int layer = node.MaxLayer; layer >= 0; layer--)
            {
                ResetNodeConnectionsAtLayer(node, layer);
            }
        }

        internal void ResetNodeConnectionsAtLayer(Node node, int layer)
        {
            node.OutEdges[layer] = new List<int>(data.MaxEdges(layer) + 1);
            node.InEdges[layer] = new List<int>(data.MaxEdges(layer) + 1);
        }

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
