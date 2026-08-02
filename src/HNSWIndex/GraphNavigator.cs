using System.Numerics;

namespace HNSWIndex
{
    internal class GraphNavigator<TVector, TDistance> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance>
    {
        private static Func<int, bool> noFilter = _ => true;
        private static Func<int, bool> noLayerFilter = (_) => true;

        private VisitedListPool pool;
        private GraphData<TVector, TDistance> data;
        private DistanceComparer<TDistance> fartherFirst;
        private ReverseDistanceComparer<TDistance> closerFirst;

        internal GraphNavigator(GraphData<TVector, TDistance> graphData)
        {
            data = graphData;
            pool = new VisitedListPool(Environment.ProcessorCount, graphData.Capacity);
            fartherFirst = new DistanceComparer<TDistance>();
            closerFirst = new ReverseDistanceComparer<TDistance>();
        }

        /// <summary>
        /// Find entry point for qury search at specified layer.
        /// Optional filter function can discriminate specific candidates. 
        /// </summary>
        internal Node FindEntryPoint(int dstLayer, TVector query, Func<int, bool>? filterFnc = null)
        {
            var bestPeer = data.EntryPoint;
            for (int layer = bestPeer.MaxLayer; layer > dstLayer; layer--)
                bestPeer = FindEntryAtLayer(layer, bestPeer, query, filterFnc);
            return bestPeer;
        }

        /// <summary>
        /// Find entry point for qury search at specified layer without locking.
        /// Optional filter function can discriminate specific candidates. 
        /// </summary>
        internal Node FindEntryPointQuery(int dstLayer, TVector query, Func<int, bool>? filterFnc = null)
        {
            var bestPeer = data.EntryPoint;
            for (int layer = bestPeer.MaxLayer; layer > dstLayer; layer--)
                bestPeer = FindEntryAtLayer(layer, bestPeer, query, filterFnc);
            return bestPeer;
        }

        /// <summary>
        /// Search for best entry point at specific layer.
        /// Filter funtion discriminates certain solution.
        /// </summary>
        internal Node FindEntryAtLayer(int layer, Node startNode, TVector query, Func<int, bool>? filterFnc = null)
        {
            filterFnc ??= noLayerFilter;

            var bestPeer = startNode;
            var bestPeerCandidate = bestPeer;
            var currDist = data.Distance(bestPeerCandidate.Id, query);

            bool changed = true;
            while (changed)
            {
                changed = false;
                lock (bestPeerCandidate.OutEdgesLock)
                {
                    var connections = bestPeerCandidate.OutEdges[layer].AsSpan();

                    for (int i = 0; i < connections.Length; i++)
                    {
                        int candidateId = connections[i];
                        var d = data.Distance(candidateId, query);
                        if (d < currDist)
                        {
                            currDist = d;
                            bestPeerCandidate = data.Nodes[candidateId];
                            if (filterFnc(candidateId)) bestPeer = bestPeerCandidate;
                            changed = true;
                        }
                    }
                }
            }
            return bestPeer;
        }

        /// <summary>
        /// Search for best entry point at specific layer without locking.
        /// Filter funtion discriminates certain solution.
        /// </summary>
        internal Node FindEntryAtLayerQuery(int layer, Node startNode, TVector query, Func<int, bool>? filterFnc = null)
        {
            filterFnc ??= noLayerFilter;

            var bestPeer = startNode;
            var bestPeerCandidate = bestPeer;
            var currDist = data.Distance(bestPeerCandidate.Id, query);

            bool changed = true;
            while (changed)
            {
                changed = false;
                var connections = bestPeerCandidate.OutEdges[layer].AsSpan();

                for (int i = 0; i < connections.Length; i++)
                {
                    int candidateId = connections[i];
                    var d = data.Distance(candidateId, query);
                    if (d < currDist)
                    {
                        currDist = d;
                        bestPeerCandidate = data.Nodes[candidateId];
                        if (filterFnc(candidateId)) bestPeer = bestPeerCandidate;
                        changed = true;
                    }
                }
            }
            return bestPeer;
        }

        /// <summary>
        /// Perform search for k closest neighbors to queryPoint at given layer.
        /// Search starts at entry point. Some points may be excluded from search with filter funcion.
        /// Search is protected against concurrent modifications.
        /// </summary>
        internal NodeDistance<TDistance>[] SearchLayer(int entryPointId, int layer, int k, TVector queryPoint, Func<int, bool>? filterFnc = null)
        {
            filterFnc ??= noFilter;
            var topCandidates = new BinaryHeap<NodeDistance<TDistance>, DistanceComparer<TDistance>>(k, fartherFirst);
            var candidates = new BinaryHeap<NodeDistance<TDistance>, ReverseDistanceComparer<TDistance>>(k * 2, closerFirst);

            var entry = new NodeDistance<TDistance>(entryPointId, data.Distance(entryPointId, queryPoint));
            var farthestResultDist = TDistance.MaxValue;

            if (filterFnc(entryPointId))
            {
                topCandidates.Push(entry);
                farthestResultDist = entry.Dist;
            }

            candidates.Push(entry);
            var visitedList = pool.GetFreeVisitedList();
            visitedList.Add(entryPointId);

            // run bfs
            while (candidates.Count > 0)
            {
                // get next candidate to check and expand
                var closestCandidate = candidates.Pop();
                if (closestCandidate.Dist > farthestResultDist && topCandidates.Count >= k)
                {
                    break;
                }

                var candidateNode = data.Nodes[closestCandidate.Id];

                lock (candidateNode.OutEdgesLock)
                {
                    var neighborsIds = candidateNode.OutEdges[layer].AsSpan();

                    for (int i = 0; i < neighborsIds.Length; ++i)
                    {
                        int neighborId = neighborsIds[i];
                        if (visitedList.Contains(neighborId)) continue;

                        var neighborDistance = data.Distance(neighborId, queryPoint);
                        // enqueue perspective neighbors to expansion list
                        if (topCandidates.Count < k || neighborDistance < farthestResultDist)
                        {
                            var selectedCandidate = new NodeDistance<TDistance>(neighborId, neighborDistance);
                            candidates.Push(selectedCandidate);

                            if (filterFnc(selectedCandidate.Id))
                                topCandidates.Push(selectedCandidate);

                            if (topCandidates.Count > k)
                                topCandidates.Pop();

                            if (topCandidates.Count > 0)
                                farthestResultDist = topCandidates.Peek().Dist;
                        }

                        // update visited list
                        visitedList.Add(neighborId);
                    }
                }
            }

            pool.ReleaseVisitedList(visitedList);

            return topCandidates.ToArray();
        }

        /// <summary>
        /// Search without protection against concurrent modifications.
        /// </summary>
        internal NodeDistance<TDistance>[] SearchLayerQuery(int entryPointId, int layer, int k, TVector queryPoint, Func<int, bool>? filterFnc = null)
        {
            filterFnc ??= noFilter;
            var topCandidates = new BinaryHeap<NodeDistance<TDistance>, DistanceComparer<TDistance>>(k, fartherFirst);
            var candidates = new BinaryHeap<NodeDistance<TDistance>, ReverseDistanceComparer<TDistance>>(k * 2, closerFirst);

            var entry = new NodeDistance<TDistance>(entryPointId, data.Distance(entryPointId, queryPoint));
            var farthestResultDist = TDistance.MaxValue;

            if (filterFnc(entryPointId))
            {
                topCandidates.Push(entry);
                farthestResultDist = entry.Dist;
            }

            candidates.Push(entry);
            var visitedList = pool.GetFreeVisitedList();
            visitedList.Add(entryPointId);

            // run bfs
            while (candidates.Count > 0)
            {
                // get next candidate to check and expand
                var closestCandidate = candidates.Pop();
                if (closestCandidate.Dist > farthestResultDist && topCandidates.Count >= k)
                {
                    break;
                }

                var candidateNode = data.Nodes[closestCandidate.Id];
                var neighborsIds = candidateNode.OutEdges[layer].AsSpan();

                for (int i = 0; i < neighborsIds.Length; ++i)
                {
                    int neighborId = neighborsIds[i];
                    if (visitedList.Contains(neighborId)) continue;

                    var neighborDistance = data.Distance(neighborId, queryPoint);
                    // enqueue perspective neighbors to expansion list
                    if (topCandidates.Count < k || neighborDistance < farthestResultDist)
                    {
                        var selectedCandidate = new NodeDistance<TDistance>(neighborId, neighborDistance);
                        candidates.Push(selectedCandidate);

                        if (filterFnc(selectedCandidate.Id))
                            topCandidates.Push(selectedCandidate);

                        if (topCandidates.Count > k)
                            topCandidates.Pop();

                        if (topCandidates.Count > 0)
                            farthestResultDist = topCandidates.Peek().Dist;
                    }

                    // update visited list
                    visitedList.Add(neighborId);
                }
            }

            pool.ReleaseVisitedList(visitedList);

            return topCandidates.ToArray();
        }

        /// <summary>
        /// Range based search for neighbors to queryPoint at given layer.
        /// Use in stateless search as it lacks locking and is not protected against concurrent modifications.
        /// </summary>
        internal NodeDistance<TDistance>[] SearchLayerRange(int entryPointId, int layer, TDistance range, TVector queryPoint, Func<int, bool>? filterFnc = null)
        {
            filterFnc ??= noFilter;
            var topCandidates = new BinaryHeap<NodeDistance<TDistance>, DistanceComparer<TDistance>>(data.MaxEdges(layer), fartherFirst);
            var candidates = new BinaryHeap<NodeDistance<TDistance>, ReverseDistanceComparer<TDistance>>(data.MaxEdges(layer) * 2, closerFirst);

            var entry = new NodeDistance<TDistance>(entryPointId, data.Distance(entryPointId, queryPoint));
            var farthestResultDist = TDistance.MaxValue;

            if (filterFnc(entryPointId) && entry.Dist <= range)
            {
                topCandidates.Push(entry);
                farthestResultDist = entry.Dist;
            }

            candidates.Push(entry);
            var visitedList = pool.GetFreeVisitedList();
            visitedList.Add(entryPointId);

            // run bfs
            while (candidates.Count > 0)
            {
                // get next candidate to check and expand
                var closestCandidate = candidates.Peek();
                if (closestCandidate.Dist > farthestResultDist && closestCandidate.Dist > range)
                {
                    break;
                }
                candidates.Pop(); // Delay heap reordering in case of early break 

                var neighborsIds = data.Nodes[closestCandidate.Id].OutEdges[layer].AsSpan();

                for (int i = 0; i < neighborsIds.Length; ++i)
                {
                    int neighborId = neighborsIds[i];
                    if (visitedList.Contains(neighborId)) continue;

                    var neighborDistance = data.Distance(neighborId, queryPoint);

                    // enqueue perspective neighbors to expansion list
                    if (neighborDistance <= range)
                    {
                        var selectedCandidate = new NodeDistance<TDistance>(neighborId, neighborDistance);
                        candidates.Push(selectedCandidate);

                        if (filterFnc(selectedCandidate.Id))
                            topCandidates.Push(selectedCandidate);

                        if (topCandidates.Peek().Dist > range)
                            topCandidates.Pop();

                        if (topCandidates.Count > 0)
                            farthestResultDist = topCandidates.Peek().Dist;
                    }

                    // update visited list
                    visitedList.Add(neighborId);
                }
            }

            pool.ReleaseVisitedList(visitedList);

            return topCandidates.ToArray();
        }

        /// <summary>
        /// Count weakly connected components at each layer.
        /// Returned array is indexed by layer id starting from zero.
        /// </summary>
        internal int[] GetConnectedComponentCounts()
        {
            if (data.Count == 0) return [];

            var activeIds = data.ActiveIdsSnapshot();
            if (activeIds.Length == 0) return [];

            int topLayer = data.GetTopLayer();
            var counts = new int[topLayer + 1];

            for (int layer = 0; layer <= topLayer; layer++)
            {
                var nodesOnLayer = activeIds.Where(id => data.Nodes[id].MaxLayer >= layer).ToArray();
                counts[layer] = CountWeaklyConnectedComponentsAtLayer(nodesOnLayer, layer);
            }

            return counts;
        }

        private int CountWeaklyConnectedComponentsAtLayer(int[] nodesOnLayer, int layer)
        {
            if (nodesOnLayer.Length == 0) return 0;

            var nodesOnLayerSet = new HashSet<int>(nodesOnLayer);
            var outgoingEdges = new Dictionary<int, int[]>(nodesOnLayer.Length);
            var incomingEdges = new Dictionary<int, List<int>>(nodesOnLayer.Length);

            for (int i = 0; i < nodesOnLayer.Length; i++)
            {
                int nodeId = nodesOnLayer[i];
                var node = data.Nodes[nodeId];

                int[] neighbors;
                lock (node.OutEdgesLock)
                {
                    neighbors = node.OutEdges[layer].ToArray();
                }

                outgoingEdges[nodeId] = neighbors;
                for (int j = 0; j < neighbors.Length; j++)
                {
                    int neighborId = neighbors[j];
                    if (!nodesOnLayerSet.Contains(neighborId)) continue;

                    if (!incomingEdges.TryGetValue(neighborId, out var incoming))
                    {
                        incoming = new List<int>();
                        incomingEdges[neighborId] = incoming;
                    }

                    incoming.Add(nodeId);
                }
            }

            int componentCount = 0;
            var visited = new HashSet<int>(nodesOnLayer.Length);
            var frontier = new Queue<int>();

            for (int i = 0; i < nodesOnLayer.Length; i++)
            {
                int startNodeId = nodesOnLayer[i];
                if (!visited.Add(startNodeId)) continue;

                componentCount++;
                frontier.Enqueue(startNodeId);

                while (frontier.Count > 0)
                {
                    int nodeId = frontier.Dequeue();

                    var outgoing = outgoingEdges[nodeId];
                    for (int j = 0; j < outgoing.Length; j++)
                    {
                        int neighborId = outgoing[j];
                        if (!nodesOnLayerSet.Contains(neighborId)) continue;
                        if (visited.Add(neighborId)) frontier.Enqueue(neighborId);
                    }

                    if (!incomingEdges.TryGetValue(nodeId, out var incoming)) continue;
                    for (int j = 0; j < incoming.Count; j++)
                    {
                        int neighborId = incoming[j];
                        if (visited.Add(neighborId)) frontier.Enqueue(neighborId);
                    }
                }
            }

            return componentCount;
        }

        internal void OnReallocate(int newCapacity)
        {
            pool.Resize(newCapacity);
        }
    }
}
