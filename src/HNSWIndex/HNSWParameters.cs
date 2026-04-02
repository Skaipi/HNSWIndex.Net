using System.Numerics;
using ProtoBuf;

namespace HNSWIndex
{
    [ProtoContract]
    public class HNSWParameters<TDistance> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance>
    {
        /// <summary>
        /// Number of outgoing edges from nodes. Number of edges on layer 0 might not obey this limit.
        /// </summary>
        [ProtoMember(1)]
        public int MaxEdges { get; set; } = 16;

        /// <summary>
        /// Rate parameter for exponential distribution.
        /// </summary>
        [ProtoMember(2)]
        public double DistributionRate { get; set; } = 1 / Math.Log(16);

        /// <summary>
        /// The minimal number of nodes obtained by knn search. If provided k exceeds this value, the search result will be trimmed to k. Improves recall for small k.
        /// </summary>
        [ProtoMember(3)]
        public int MinNN { get; set; } = 5;

        /// <summary>
        /// Maximum number of nodes taken as candidates for neighbor check during insertion
        /// </summary>
        [ProtoMember(4)]
        public int MaxCandidates { get; set; } = 100;

        /// <summary>
        /// Maximum number of nodes taken as candidates for neighbor check during removal
        /// </summary>
        [ProtoMember(5)]
        public int RemoveMaxCandidates { get; set; } = 150;

        /// <summary>
        /// Expected amount of nodes in the graph.
        /// </summary>
        [ProtoMember(6)]
        public int CollectionSize { get; set; } = 65536;

        /// <summary>
        /// Seed for RNG. Values below 0 are taken as no seed.
        /// </summary>
        [ProtoMember(7)]
        public int RandomSeed { get; set; } = 31337;

        /// <summary>
        /// Indicates if removals are allowed in the index. Setting this to false improves parallelization performance.
        /// </summary>
        [ProtoMember(8)]
        public bool AllowRemovals = true;
    }
}