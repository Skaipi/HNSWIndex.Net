using System.Numerics;

namespace HNSWIndex
{
    internal static class Heuristic<TDistance> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance>
    {
        internal static DistanceComparer<TDistance> FartherFirst = new DistanceComparer<TDistance>();
        internal static ReverseDistanceComparer<TDistance> CloserFirst = new ReverseDistanceComparer<TDistance>();

        internal static List<int> DefaultHeuristic(NodeDistance<TDistance>[] candidates, Func<int, int, TDistance> distanceFnc, int maxEdges)
        {
            if (candidates.Length < maxEdges)
            {
                var ids = new List<int>(candidates.Length);
                for (int i = 0; i < candidates.Length; i++) ids.Add(candidates[i].Id);
                return ids;
            }

            var resultList = new List<NodeDistance<TDistance>>(maxEdges + 1);
            Array.Sort(candidates, FartherFirst); // FarherFirst applies to heap ordering not sorting
            for (int i = 0; i < candidates.Length && resultList.Count < maxEdges; i++)
            {
                // Make local copy to maybe improve performance
                var candidate = candidates[i];
                var candidateId = candidate.Id;
                var candidateDist = candidate.Dist;

                bool acceptable = true;
                for (int j = 0; j < resultList.Count; j++)
                {
                    var s = resultList[j];
                    if (distanceFnc(s.Id, candidateId) < candidateDist) { acceptable = false; break; }
                }

                if (acceptable) resultList.Add(candidate);
            }

            var outIds = new List<int>(resultList.Count);
            for (int k = 0; k < resultList.Count; k++) outIds.Add(resultList[k].Id);
            return outIds;
        }
    }
}
