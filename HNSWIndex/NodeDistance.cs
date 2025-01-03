using System.Numerics;

namespace HNSWIndex
{
    public struct NodeDistance<TDistance> where TDistance : struct, IFloatingPoint<TDistance>
    {
        public int Id { get; set; }
        public TDistance Dist { get; set; }
    }
}