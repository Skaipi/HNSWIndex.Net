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

        internal void RemoveConnectionsAtLayer(Node removedNode, int layer)
        {
            if (removedNode.Id == data.EntryPointId)
            {
                var replacementFound = data.TryReplaceEntryPoint(layer);
                if (!replacementFound && layer == 0)
                {
                    if (data.Nodes.Count > 0) throw new InvalidOperationException("Delete on isolated enry point");
                    data.EntryPointId = -1;
                }
            }

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
            // Locks should be acquired beforehand. 
            target.OutEdges[layer].Remove(badNeighbour.Id);
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
