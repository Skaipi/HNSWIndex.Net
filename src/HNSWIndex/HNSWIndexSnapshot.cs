using System.Numerics;
using ProtoBuf;

namespace HNSWIndex
{
    /// <summary>
    /// Wrapper for HNSWIndex for serialization.
    /// </summary>
    [ProtoContract]
    internal class HNSWIndexSnapshot<TLabel, TDistance> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance>
    {
        [ProtoMember(1)]
        internal HNSWParameters<TDistance>? Parameters { get; set; }

        [ProtoMember(2)]
        internal GraphDataSnapshot<TLabel, TDistance>? DataSnapshot { get; set; }

        internal HNSWIndexSnapshot() { }

        internal HNSWIndexSnapshot(HNSWParameters<TDistance> parameters, GraphData<TLabel, TDistance> data)
        {
            Parameters = parameters;
            DataSnapshot = new GraphDataSnapshot<TLabel, TDistance>(data);
        }
    }
}