using System.Collections;
using System.Numerics;
using ProtoBuf;

namespace HNSWIndex
{
    /// <summary>
    /// Wrapper for GraphData for serialization.
    /// </summary>
    [ProtoContract]
    internal class GraphDataSnapshot<TLabel, TDistance> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance>
    {
        [ProtoMember(1)]
        internal Node[]? Nodes { get; set; }

        [ProtoMember(2)]
        internal NestedArrayWrapper<TLabel>[]? Items { get; set; }

        [ProtoMember(3)]
        internal Queue<int>? RemovedIndexes { get; set; }

        [ProtoMember(4)]
        internal int EntryPointId = -1;

        [ProtoMember(5)]
        internal int Capacity;

        [ProtoMember(6)]
        internal int Length;

        [ProtoMember(7)]
        internal int Count;

        internal TLabel[]? ParsedItems
        {
            get
            {
                var items = Items?.Select(i => i.Values).ToArray();
                Array.Resize(ref items, Capacity);
                return items;
            }
        }

        internal Node[]? ParsedNodes
        {
            get
            {
                var nodes = Nodes;
                Array.Resize(ref nodes, Capacity);
                return nodes;
            }
        }

        internal GraphDataSnapshot() { }

        internal GraphDataSnapshot(GraphData<TLabel, TDistance> data)
        {
            Nodes = data.Nodes.Where(n => n is not null).ToArray();
            Items = data.Items.Where(i => i is not null).Select(i => new NestedArrayWrapper<TLabel>(i)).ToArray();
            RemovedIndexes = data.RemovedIndexes;
            EntryPointId = data.EntryPointId;
            Capacity = data.Capacity;
            Length = data.Length;
            Count = data.Count;
        }
    }
}