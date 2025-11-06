using System.Numerics;

namespace HNSWIndex
{
    internal class GraphNavigator<TLabel, TDistance> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance>
    {
        private static Func<int, bool> noFilter = _ => true;
        private static Func<int, bool> noLayerFilter = (_) => true;

        private VisitedListPool pool;
        private GraphData<TLabel, TDistance> data;
        private DistanceComparer<TDistance> fartherFirst;
        private ReverseDistanceComparer<TDistance> closerFirst;

        internal GraphNavigator(GraphData<TLabel, TDistance> graphData)
        {
            data = graphData;
            pool = new VisitedListPool(1, graphData.Capacity);
            fartherFirst = new DistanceComparer<TDistance>();
            closerFirst = new ReverseDistanceComparer<TDistance>();
        }

        /// <summary>
        /// Find entry point for qury search at specified layer.
        /// Default locking is in writer mode and can be changed.
        /// Optional filter function can discriminate specific candidates. 
        /// </summary>
        internal Node FindEntryPoint(int dstLayer, TLabel query, Func<int, bool>? filterFnc = null)
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
        internal Node FindEntryAtLayer(int layer, Node startNode, TLabel query, Func<int, bool>? filterFnc = null)
        {
            filterFnc ??= noLayerFilter;

            var bestPeer = startNode;
            var bestPeerCandidate = bestPeer;
            var currDist = data.Distance(bestPeerCandidate.Id, query);

            bool changed = true;
            while (changed)
            {
                changed = false;
                List<int> connections = bestPeerCandidate.OutEdges[layer];
                int size = connections.Count;

                for (int i = 0; i < size; i++)
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
        /// Default lock in this method is in writer mode.
        /// </summary>
        internal NodeDistance<TDistance>[] SearchLayer(int entryPointId, int layer, int k, TLabel queryPoint, Func<int, bool>? filterFnc = null)
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
                var closestCandidate = candidates.Peek();
                if (closestCandidate.Dist > farthestResultDist && topCandidates.Count >= k)
                {
                    break;
                }
                candidates.Pop(); // Delay heap reordering in case of early break 

                var neighboursIds = data.Nodes[closestCandidate.Id].OutEdges[layer];

                for (int i = 0; i < neighboursIds.Count; ++i)
                {
                    int neighbourId = neighboursIds[i];
                    if (visitedList.Contains(neighbourId)) continue;

                    var neighbourDistance = data.Distance(neighbourId, queryPoint);

                    // enqueue perspective neighbours to expansion list
                    if (topCandidates.Count < k || neighbourDistance < farthestResultDist)
                    {
                        var selectedCandidate = new NodeDistance<TDistance>(neighbourId, neighbourDistance);
                        candidates.Push(selectedCandidate);

                        if (filterFnc(selectedCandidate.Id))
                            topCandidates.Push(selectedCandidate);

                        if (topCandidates.Count > k)
                            topCandidates.Pop();

                        if (topCandidates.Count > 0)
                            farthestResultDist = topCandidates.Peek().Dist;
                    }

                    // update visited list
                    visitedList.Add(neighbourId);
                }
            }

            pool.ReleaseVisitedList(visitedList);

            return topCandidates.ToArray();
        }

        // TODO: Merge this method with SearchLayer
        internal NodeDistance<TDistance>[] SearchLayerRange(int entryPointId, int layer, TDistance range, TLabel queryPoint, Func<int, bool>? filterFnc = null)
        {
            filterFnc ??= noFilter;
            var topCandidates = new BinaryHeap<NodeDistance<TDistance>, DistanceComparer<TDistance>>(data.MaxEdges(layer), fartherFirst);
            var candidates = new BinaryHeap<NodeDistance<TDistance>, ReverseDistanceComparer<TDistance>>(data.MaxEdges(layer) * 2, closerFirst);

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
                var closestCandidate = candidates.Peek();
                if (closestCandidate.Dist > farthestResultDist && closestCandidate.Dist > range)
                {
                    break;
                }
                candidates.Pop(); // Delay heap reordering in case of early break 

                var neighboursIds = data.Nodes[closestCandidate.Id].OutEdges[layer];

                for (int i = 0; i < neighboursIds.Count; ++i)
                {
                    int neighbourId = neighboursIds[i];
                    if (visitedList.Contains(neighbourId)) continue;

                    var neighbourDistance = data.Distance(neighbourId, queryPoint);

                    // enqueue perspective neighbours to expansion list
                    if (neighbourDistance <= range)
                    {
                        var selectedCandidate = new NodeDistance<TDistance>(neighbourId, neighbourDistance);
                        candidates.Push(selectedCandidate);

                        if (filterFnc(selectedCandidate.Id))
                            topCandidates.Push(selectedCandidate);

                        if (topCandidates.Peek().Dist > range)
                            topCandidates.Pop();

                        if (topCandidates.Count > 0)
                            farthestResultDist = topCandidates.Peek().Dist;
                    }

                    // update visited list
                    visitedList.Add(neighbourId);
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