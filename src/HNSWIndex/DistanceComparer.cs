using System.Numerics;
using System.Runtime.CompilerServices;

namespace HNSWIndex
{
    internal struct DistanceComparer<TDistance> : IComparer<NodeDistance<TDistance>> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Compare(NodeDistance<TDistance> x, NodeDistance<TDistance> y)
        {
            if (x.Dist < y.Dist) return -1;
            if (x.Dist > y.Dist) return 1;
            return x.Dist.CompareTo(y.Dist);
        }
    }

    internal struct ReverseDistanceComparer<TDistance> : IComparer<NodeDistance<TDistance>> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Compare(NodeDistance<TDistance> x, NodeDistance<TDistance> y)
        {
            if (x.Dist > y.Dist) return -1;
            if (x.Dist < y.Dist) return 1;
            return y.Dist.CompareTo(x.Dist);
        }
    }
}