using System.Numerics;
using System.Runtime.CompilerServices;

namespace HNSWIndex
{
    public static class Heuristic<TDistance> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance>
    {
        internal static DistanceComparer<TDistance> FartherFirst = new DistanceComparer<TDistance>();
        internal static ReverseDistanceComparer<TDistance> CloserFirst = new ReverseDistanceComparer<TDistance>();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static EdgeList RelativeNeighborPruning(NodeDistance<TDistance>[] candidates, Func<int, int, TDistance> distanceFnc, int maxEdges)
        {
            if (candidates.Length < maxEdges)
            {
                var ids = new EdgeList(candidates.Length);
                for (int i = 0; i < candidates.Length; i++) ids.Add(candidates[i].Id);
                return ids;
            }

            var resultCount = 0;
            var resultList = new NodeDistance<TDistance>[maxEdges + 1];
            Array.Sort(candidates, FartherFirst); // FarherFirst applies to heap ordering not sorting
            for (int i = 0; i < candidates.Length && resultCount < maxEdges; i++)
            {
                // Make local copy to maybe improve performance
                var candidate = candidates[i];
                var candidateId = candidate.Id;
                var candidateDist = candidate.Dist;

                bool acceptable = true;
                for (int j = 0; j < resultCount; j++)
                {
                    var s = resultList[j];
                    if (distanceFnc(s.Id, candidateId) < candidateDist) { acceptable = false; break; }
                }

                if (acceptable)
                {
                    resultList[resultCount++] = candidate;
                }
            }

            var outIds = new EdgeList(maxEdges + 1);
            for (int k = 0; k < resultCount; k++) outIds.Add(resultList[k].Id);
            return outIds;
        }
    }
}
