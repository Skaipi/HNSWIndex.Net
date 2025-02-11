using System.Numerics;

namespace HNSWIndex
{
    internal class GraphConnector<TLabel, TDistance> where TDistance : struct, IFloatingPoint<TDistance>
    {
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
                data.SetEntryPoint(nodeId);
                Monitor.Exit(data.entryPointLock);
                return;
            }

            var currNode = data.Nodes[nodeId];
            if (currNode.MaxLayer > data.GetTopLayer())
            {
                AddNewConnections(currNode);
                data.SetEntryPoint(nodeId);
                Monitor.Exit(data.entryPointLock);
            }
            else
            {
                Monitor.Exit(data.entryPointLock);
                AddNewConnections(currNode);
            }
        }

        internal void RemoveConnections(int itemIndex)
        {
            var node = data.Nodes[itemIndex];
            lock (node.OutEdgesLock)
            {
                for (int layer = 0; layer <= node.MaxLayer; layer++)
                {
                    var mOnLayer = data.MaxEdges(layer);
                    // NOTE: each candidate loose exactly one in-edge
                    for (int j = 0; j < node.OutEdges[layer].Count; j++)
                    {
                        var neighbourId = node.OutEdges[layer][j];
                        var neighbourNode = data.Nodes[neighbourId];
                        RemoveInEdge(neighbourNode, node, layer);
                    }

                    var orphanNodes = node.OutEdges[layer];
                    for (int j = 0; j < node.InEdges[layer].Count; j++)
                    {
                        var neighbourId = node.InEdges[layer][j];
                        var neighbourNode = data.Nodes[neighbourId];
                        RemoveOutEdge(neighbourNode, node, layer);

                        var distanceCalculator = new DistanceCalculator<int, TDistance>(data.Distance, neighbourId);
                        var candidates = new List<NodeDistance<TDistance>>();
                        for (int i = 0; i < orphanNodes.Count; i++)
                        {
                            int candidateId = orphanNodes[i];
                            if (candidateId == neighbourId)
                                continue;

                            if (!neighbourNode.OutEdges[layer].Contains(candidateId))
                            {
                                candidates.Add(new NodeDistance<TDistance>
                                {
                                    Dist = distanceCalculator.From(candidateId),
                                    Id = candidateId
                                });
                            }
                        }
                        var selectedCandidates = parameters.Heuristic(candidates, data.Distance, mOnLayer);

                        for (int i = 0; i < selectedCandidates.Count; i++)
                        {
                            int newNeighbourId = selectedCandidates[i];
                            var newNeighbour = data.Nodes[newNeighbourId];
                            Connect(neighbourNode, newNeighbour, layer);

                            if (!newNeighbour.OutEdges[layer].Contains(neighbourId))
                            {
                                Connect(newNeighbour, neighbourNode, layer);
                            }

                            orphanNodes.Remove(newNeighbourId);
                        }
                    }
                }
            }
        }

        internal void RecomputeNodeConnectionsAtLayer(int nodeId, int layer, Node src)
        {
            var currentNode = data.Nodes[nodeId];
            var distanceCalculator = new DistanceCalculator<int, TDistance>(data.Distance, nodeId);
            var bestPeer = navigator.FindEntryPoint(layer, distanceCalculator);

            var topCandidates = navigator.SearchLayer(bestPeer.Id, layer, parameters.MaxCandidates, distanceCalculator, id => id != nodeId && id != src.Id);
            var bestNeighboursIds = parameters.Heuristic(topCandidates, data.Distance, data.MaxEdges(layer));

            for (int i = 0; i < bestNeighboursIds.Count; ++i)
            {
                int newNeighbourId = bestNeighboursIds[i];
                Connect(currentNode, data.Nodes[newNeighbourId], layer);
                Connect(data.Nodes[newNeighbourId], currentNode, layer);
            }
        }

        internal void RemoveOutEdge(Node target, Node badNeighbour, int layer)
        {
            lock (target.OutEdgesLock)
            {
                target.OutEdges[layer].Remove(badNeighbour.Id);
            }
        }

        internal void RemoveInEdge(Node target, Node badNeighbour, int layer)
        {
            lock (target.InEdgesLock)
            {
                target.InEdges[layer].Remove(badNeighbour.Id);
            }
        }

        private void AddNewConnections(Node currNode)
        {
            var distCalculator = new DistanceCalculator<int, TDistance>(data.Distance, currNode.Id);
            var bestPeer = navigator.FindEntryPoint(currNode.MaxLayer, distCalculator);

            for (int layer = Math.Min(currNode.MaxLayer, data.GetTopLayer()); layer >= 0; --layer)
            {
                var topCandidates = navigator.SearchLayer(bestPeer.Id, layer, parameters.MaxCandidates, distCalculator);
                var bestNeighboursIds = parameters.Heuristic(topCandidates, data.Distance, data.MaxEdges(layer));

                for (int i = 0; i < bestNeighboursIds.Count; ++i)
                {
                    int newNeighbourId = bestNeighboursIds[i];
                    Connect(currNode, data.Nodes[newNeighbourId], layer);
                    Connect(data.Nodes[newNeighbourId], currNode, layer);
                }
            }
        }

        private void Connect(Node node, Node neighbour, int layer)
        {
            lock (node.OutEdgesLock)
            {
                node.OutEdges[layer].Add(neighbour.Id);
                lock (neighbour.InEdgesLock)
                {
                    neighbour.InEdges[layer].Add(node.Id);
                }
                if (node.OutEdges[layer].Count > data.MaxEdges(layer))
                {
                    foreach (var neighbourId in node.OutEdges[layer])
                    {
                        lock (data.Nodes[neighbourId].InEdgesLock)
                        {
                            data.Nodes[neighbourId].InEdges[layer].Remove(node.Id);
                        }
                    }

                    var candidates = new List<NodeDistance<TDistance>>(node.OutEdges[layer].Count);
                    foreach (var neighbourId in node.OutEdges[layer])
                    {
                        candidates.Add(new NodeDistance<TDistance> { Dist = data.Distance(neighbourId, node.Id), Id = neighbourId });
                    }

                    var selectedCandidates = parameters.Heuristic(candidates, data.Distance, data.MaxEdges(layer));
                    node.OutEdges[layer] = selectedCandidates;

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
}
