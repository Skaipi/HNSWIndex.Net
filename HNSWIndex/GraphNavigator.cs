using System.Collections;
using System.Numerics;

namespace HNSWIndex
{
    internal class GraphNavigator<TLabel, TDistance> where TDistance : struct, IFloatingPoint<TDistance> where TLabel : IList
    {
        private static Func<int, bool> noFilter = _ => true;

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

        internal Node FindEntryPoint<T>(int dstLayer, DistanceCalculator<T, TDistance> dstDistance, bool locking = true)
        {
            var bestPeer = data.EntryPoint;
            var currDist = dstDistance.From(bestPeer.Id);

            for (int level = bestPeer.MaxLayer; level > dstLayer; level--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    using var _ = new OptionalLock(locking, data.Nodes[bestPeer.Id].OutEdgesLock);
                    List<int> connections = bestPeer.OutEdges[level];
                    int size = connections.Count;

                    for (int i = 0; i < size; i++)
                    {
                        int cand = connections[i];
                        var d = dstDistance.From(cand);
                        if (d < currDist)
                        {
                            currDist = d;
                            bestPeer = data.Nodes[cand];
                            changed = true;
                        }
                    }
                }
            }

            return bestPeer;
        }

        internal Node FindEntryAtLayer<T>(int layer, Node startNode, DistanceCalculator<T, TDistance> dstDistance, bool locking = true)
        {
            // TODO: Check if this logic can be extracted to regular FindEndtryPoint
            var bestPeer = startNode;
            var currDist = dstDistance.From(bestPeer.Id);
            bool changed = true;
            while (changed)
            {
                changed = false;
                using var _ = new OptionalLock(locking, data.Nodes[bestPeer.Id].OutEdgesLock);
                List<int> connections = bestPeer.OutEdges[layer];
                int size = connections.Count;

                for (int i = 0; i < size; i++)
                {
                    int cand = connections[i];
                    var d = dstDistance.From(cand);
                    if (d < currDist)
                    {
                        currDist = d;
                        bestPeer = data.Nodes[cand];
                        changed = true;
                    }
                }
            }
            return bestPeer;
        }

        internal List<NodeDistance<TDistance>> SearchLayer<T>(int entryPointId, int layer, int k, DistanceCalculator<T, TDistance> distanceCalculator, Func<int, bool>? filterFnc = null, bool locking = true)
        {
            filterFnc ??= noFilter;
            var topCandidates = new BinaryHeap<NodeDistance<TDistance>>(new List<NodeDistance<TDistance>>(k), fartherFirst);
            var candidates = new BinaryHeap<NodeDistance<TDistance>>(new List<NodeDistance<TDistance>>(k * 2), closerFirst); // Guess that k*2 space is usually enough

            var entry = new NodeDistance<TDistance> { Dist = distanceCalculator.From(entryPointId), Id = entryPointId };
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
                using var _ = new OptionalLock(locking, data.Nodes[closestCandidate.Id].OutEdgesLock);
                var neighboursIds = data.Nodes[closestCandidate.Id].OutEdges[layer];

                for (int i = 0; i < neighboursIds.Count; ++i)
                {
                    int neighbourId = neighboursIds[i];
                    if (visitedList.Contains(neighbourId)) continue;

                    var neighbourDistance = distanceCalculator.From(neighbourId);

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

            pool.ReleaseVisitedList(visitedList);

            return topCandidates.Buffer;
        }

        internal void OnReallocate(int newCapacity)
        {
            pool.Resize(newCapacity);
        }
    }
}