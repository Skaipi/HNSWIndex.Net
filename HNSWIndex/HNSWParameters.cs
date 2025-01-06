using System.Numerics;

namespace HNSWIndex
{
    public class HNSWParameters<TDistance> where TDistance : struct, IFloatingPoint<TDistance>
    {
        /// <summary>
        /// Number of outgoing edges from nodes. Number of edges on layer 0 might not obey this limit.
        /// </summary>
        public int MaxEdges { get; set; } = 16;

        /// <summary>
        /// Rate parameter for exponential distribution.
        /// </summary>
        public double DistributionRate { get; set; } = 1 / Math.Log(16);

        /// <summary>
        /// The minimal number of nodes obtained by knn search. If provided k exceeds this value, the search result will be trimmed to k. Improves recall for small k.
        /// </summary>
        public int MinNN { get; set; } = 5;

        /// <summary>
        /// Maximum number of nodes taken as candidates for neighbour check during insertion
        /// </summary>
        public int MaxCandidates { get; set; } = 100;

        /// <summary>
        /// Expected amount of nodes in the graph.
        /// </summary>
        public int CollectionSize { get; set; } = 65536;

        /// <summary>
        /// Seed for RNG. Values below 0 are taken as no seed.
        /// </summary>
        public int RandomSeed { get; set; } = 31337;

        /// <summary>
        /// Heuristic function for neighbour selection. This function takes a list of candidates, a distance function and a maximum number of edges to return.
        /// </summary>
        public Func<List<NodeDistance<TDistance>>, Func<int, int, TDistance>, int, List<int>> Heuristic { get; set; } = Heuristic<TDistance>.DefaultHeuristic;
    }
}