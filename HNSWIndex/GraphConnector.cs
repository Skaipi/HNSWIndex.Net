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
            for (int layer_id = 0; layer_id <= node.MaxLayer; layer_id++)
            {
                var mOnLayer = data.MaxEdges(layer_id);
                for (int j = 0; j < node.OutEdges[layer_id].Count; j++)
                {
                    var neighbourId = node.OutEdges[layer_id][j];
                    var neighbourNode = data.Nodes[neighbourId];

                    RemoveInEdge(neighbourNode, node, layer_id);
                    if (neighbourNode.OutEdges[layer_id].Count <= mOnLayer / 2 || neighbourNode.InEdges[layer_id].Count <= mOnLayer / 2)
                    {
                        RecomputeNodeConnectionsAtLayer(neighbourId, layer_id, node);
                    }
                }

                for (int j = 0; j < node.InEdges[layer_id].Count; j++)
                {
                    var neighbourId = node.InEdges[layer_id][j];
                    var neighbourNode = data.Nodes[neighbourId];
                    RemoveOutEdge(neighbourNode, node, layer_id);
                    if (neighbourNode.OutEdges[layer_id].Count <= mOnLayer / 2 || neighbourNode.InEdges[layer_id].Count <= mOnLayer / 2)
                    {
                        RecomputeNodeConnectionsAtLayer(neighbourId, layer_id, node);
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
            var bestNeighboursIds = Heuristic<TDistance>.DefaultHeuristic(topCandidates, data.Distance, data.MaxEdges(layer));

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
                var bestNeighboursIds = Heuristic<TDistance>.DefaultHeuristic(topCandidates, data.Distance, data.MaxEdges(layer));

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

                    var selectedCandidates = Heuristic<TDistance>.DefaultHeuristic(candidates, data.Distance, data.MaxEdges(layer));
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
