using System.Numerics;
using ProtoBuf;

namespace HNSWIndex
{
    /// <summary>
    /// Wrapper for HNSWIndex for serialization.
    /// </summary>
    [ProtoContract]
    internal class HNSWIndexSnapshot<TVector, TDistance> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance>
    {
        [ProtoMember(1)]
        internal HNSWParameters<TDistance>? Parameters { get; set; }

        [ProtoMember(2)]
        internal GraphDataSnapshot<TVector, TDistance>? DataSnapshot { get; set; }

        internal HNSWIndexSnapshot() { }

        internal HNSWIndexSnapshot(HNSWParameters<TDistance> parameters, GraphData<TVector, TDistance> data)
        {
            Parameters = parameters;
            DataSnapshot = new GraphDataSnapshot<TVector, TDistance>(data);
        }
    }
}