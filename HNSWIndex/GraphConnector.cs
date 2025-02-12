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
            var removedNode = data.Nodes[itemIndex];
            lock (removedNode.OutEdgesLock)
            {
                for (int layer = 0; layer <= removedNode.MaxLayer; layer++)
                {
                    WipeRelationsWithNode(removedNode, layer);

                    var mOnLayer = data.MaxEdges(layer);
                    var children = removedNode.OutEdges[layer];
                    for (int j = 0; j < removedNode.InEdges[layer].Count; j++)
                    {
                        var activeNodeId = removedNode.InEdges[layer][j];
                        var activeNode = data.Nodes[activeNodeId];
                        var activeNodedEdges = activeNode.OutEdges[layer];
                        RemoveOutEdge(activeNode, removedNode, layer);
                        WipeRelationsWithNode(activeNode, layer);

                        // Get valid candidates from children of removed node
                        var candidates = new List<int>(children.Count);
                        for (int i = 0; i < children.Count; i++)
                        {
                            int candidateId = children[i];
                            if (candidateId == activeNodeId)
                                continue;

                            if (!activeNodedEdges.Contains(candidateId))
                                candidates.Add(candidateId);
                        }

                        RecomputeConnections(activeNode, candidates.Concat(activeNodedEdges).ToList(), layer);
                        SetRelationsWithNode(activeNode, layer);
                    }
                }
            }
        }

        private void RemoveOutEdge(Node target, Node badNeighbour, int layer)
        {
            lock (target.OutEdgesLock)
            {
                target.OutEdges[layer].Remove(badNeighbour.Id);
            }
        }

        private void RemoveInEdge(Node target, Node badNeighbour, int layer)
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

        private void RecomputeConnections(Node node, List<int> candidates, int layer)
        {
            var candidatesDistances = new List<NodeDistance<TDistance>>(candidates.Count);
            foreach (var neighbourId in candidates)
                candidatesDistances.Add(new NodeDistance<TDistance> { Dist = data.Distance(neighbourId, node.Id), Id = neighbourId });
            var newNeighbours = parameters.Heuristic(candidatesDistances, data.Distance, data.MaxEdges(layer));
            node.OutEdges[layer] = newNeighbours;
        }

        private void WipeRelationsWithNode(Node node, int layer)
        {
            foreach (var neighbourId in node.OutEdges[layer])
            {
                lock (data.Nodes[neighbourId].InEdgesLock)
                {
                    data.Nodes[neighbourId].InEdges[layer].Remove(node.Id);
                }
            }
        }

        private void SetRelationsWithNode(Node node, int layer)
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
