using System.Collections.Concurrent;
using System.Numerics;
using ProtoBuf;

namespace HNSWIndex
{
    /// <summary>
    /// Wrapper for GraphData for serialization.
    /// </summary>
    [ProtoContract]
    internal class GraphDataSnapshot<TLabel, TDistance> where TDistance : struct, IFloatingPoint<TDistance>
    {
        [ProtoMember(1)]
        internal List<Node>? Nodes { get; set; }

        [ProtoMember(2)]
        internal ConcurrentDictionary<int, TLabel>? Items { get; set; }

        [ProtoMember(3)]
        internal Queue<int>? RemovedIndexes { get; set; }

        [ProtoMember(4)]
        internal int EntryPointId = -1;

        [ProtoMember(5)]
        internal int Capacity;

        internal GraphDataSnapshot() { }

        internal GraphDataSnapshot(GraphData<TLabel, TDistance> data)
        {
            Nodes = data.Nodes;
            Items = data.Items;
            RemovedIndexes = data.RemovedIndexes;
            EntryPointId = data.EntryPointId;
            Capacity = data.Capacity;
        }
    }
}