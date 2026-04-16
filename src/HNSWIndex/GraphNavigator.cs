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
        /// Default locking is in writer mode and can be changed.
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

        internal void OnReallocate(int newCapacity)
        {
            pool.Resize(newCapacity);
        }
    }
}