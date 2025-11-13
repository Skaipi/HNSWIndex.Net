using System.Numerics;

namespace HNSWIndex
{
    public readonly struct NodeDistance<TDistance> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance>
    {
        public readonly int Id;
        public readonly TDistance Dist;

        public NodeDistance(int id, TDistance distance)
        {
            Id = id;
            Dist = distance;
        }
    }
}