using System.Numerics;

namespace HNSWIndex
{
    internal static class Heuristic<TDistance> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance>
    {
        internal static IComparer<NodeDistance<TDistance>> FartherFirst = new DistanceComparer<TDistance>();
        internal static IComparer<NodeDistance<TDistance>> CloserFirst = new ReverseDistanceComparer<TDistance>();

        internal static List<int> DefaultHeuristic(List<NodeDistance<TDistance>> candidates, Func<int, int, TDistance> distanceFnc, int maxEdges)
        {
            if (candidates.Count < maxEdges)
            {
                return candidates.ConvertAll(x => x.Id);
            }

            var resultList = new List<NodeDistance<TDistance>>(maxEdges + 1);
            var candidatesHeap = new BinaryHeap<NodeDistance<TDistance>>(candidates, CloserFirst);

            while (candidatesHeap.Count > 0)
            {
                if (resultList.Count >= maxEdges)
                    break;

                var currentCandidate = candidatesHeap.Pop();
                var candidateDist = currentCandidate.Dist;

                // Candidate is closer to designated point than any other already connected point
                if (resultList.TrueForAll(connectedNode => distanceFnc(connectedNode.Id, currentCandidate.Id) > candidateDist))
                {
                    resultList.Add(currentCandidate);
                }
            }

            return resultList.ConvertAll(x => x.Id);
        }
    }
}
