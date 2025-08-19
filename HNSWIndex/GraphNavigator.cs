using System.Collections;
using System.Numerics;

namespace HNSWIndex
{
    internal class GraphNavigator<TLabel, TDistance> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance> where TLabel : IList
    {
        private static Func<int, bool> noFilter = _ => true;
        private static Func<int, bool> noLayerFilter = (_) => true;

        private VisitedListPool pool;
        private GraphData<TLabel, TDistance> data;
        private IComparer<NodeDistance<TDistance>> fartherFirst;
        private IComparer<NodeDistance<TDistance>> closerFirst;

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
        internal Node FindEntryPoint(int dstLayer, TLabel query, bool locking = true, Func<int, bool>? filterFnc = null)
        {
            var bestPeer = data.EntryPoint;
            for (int level = bestPeer.MaxLayer; level > dstLayer; level--)
                bestPeer = FindEntryAtLayer(level, bestPeer, query, locking, filterFnc);
            return bestPeer;
        }

        /// <summary>
        /// Search for best entry point at specific layer.
        /// Filter funtion discriminates certain solution.
        /// </summary>
        internal Node FindEntryAtLayer(int layer, Node startNode, TLabel query, bool locking = true, Func<int, bool>? filterFnc = null)
        {
            filterFnc ??= noLayerFilter;

            var bestPeer = startNode;
            var bestPeerCandidate = bestPeer;
            var currDist = data.Distance(bestPeerCandidate.Id, query);

            bool changed = true;
            while (changed)
            {
                changed = false;
                var shouldLock = locking && filterFnc(bestPeerCandidate.Id); // Rejected by filter are read only
                using (new OptionalLock(shouldLock, bestPeerCandidate.OutEdgesLock))
                {
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
            }
            return bestPeer;
        }

        /// <summary>
        /// Perform search for k closest neighbors to queryPoint at given layer.
        /// Search starts at entry point. Some points may be excluded from search with filter funcion.
        /// Default lock in this method is in writer mode.
        /// </summary>
        internal List<NodeDistance<TDistance>> SearchLayer(int entryPointId, int layer, int k, TLabel queryPoint, Func<int, bool>? filterFnc = null, bool locking = true)
        {
            filterFnc ??= noFilter;
            var topCandidates = new BinaryHeap<NodeDistance<TDistance>>(new List<NodeDistance<TDistance>>(k), fartherFirst);
            var candidates = new BinaryHeap<NodeDistance<TDistance>>(new List<NodeDistance<TDistance>>(k * 2), closerFirst); // Guess that k*2 space is usually enough

            var entry = new NodeDistance<TDistance> { Dist = data.Distance(entryPointId, queryPoint), Id = entryPointId };
            // TODO: Make it max value of TDistance
            var farthestResultDist = entry.Dist;

            if (filterFnc(entryPointId))
            {
                topCandidates.Push(entry);
                farthestResultDist = entry.Dist;
            }

            candidates.Push(entry);
            var visitedList = pool.GetFreeVisitedList();
            visitedList.Add(entryPointId);

            // run bfs
            while (candidates.Buffer.Count > 0)
            {
                // get next candidate to check and expand
                var closestCandidate = candidates.Buffer[0];
                if (closestCandidate.Dist > farthestResultDist && topCandidates.Count >= k)
                {
                    break;
                }
                candidates.Pop(); // Delay heap reordering in case of early break 

                // take lock if needed and expand candidate
                using (new OptionalLock(locking, data.Nodes[closestCandidate.Id].OutEdgesLock))
                {
                    var neighboursIds = data.Nodes[closestCandidate.Id].OutEdges[layer];

                    for (int i = 0; i < neighboursIds.Count; ++i)
                    {
                        int neighbourId = neighboursIds[i];
                        if (visitedList.Contains(neighbourId)) continue;

                        var neighbourDistance = data.Distance(neighbourId, queryPoint);

                        // enqueue perspective neighbours to expansion list
                        if (topCandidates.Count < k || neighbourDistance < farthestResultDist)
                        {
                            var selectedCandidate = new NodeDistance<TDistance> { Dist = neighbourDistance, Id = neighbourId };
                            candidates.Push(selectedCandidate);

                            if (filterFnc(selectedCandidate.Id))
                                topCandidates.Push(selectedCandidate);

                            if (topCandidates.Count > k)
                                topCandidates.Pop();

                            if (topCandidates.Count > 0)
                                farthestResultDist = topCandidates.Buffer[0].Dist;
                        }

                        // update visited list
                        visitedList.Add(neighbourId);
                    }
                }
            }

            pool.ReleaseVisitedList(visitedList);

            return topCandidates.Buffer;
        }

        internal void OnReallocate(int newCapacity)
        {
            pool.Resize(newCapacity);
        }
    }
}