using System.Collections.Concurrent;
using System.Numerics;
using ProtoBuf;

namespace HNSWIndex
{
    /// <summary>
    /// Wrapper for GraphData for serialization.
    /// </summary>
    [ProtoContract]
    internal class GraphDataSnapshot<TVector, TDistance> where TDistance : struct, INumber<TDistance>, IMinMaxValue<TDistance>
    {
        [ProtoMember(1)]
        internal Node[]? Nodes { get; set; }

        [ProtoMember(2)]
        internal int[]? ActiveNodes { get; set; }

        [ProtoMember(3)]
        internal NestedArrayWrapper<TVector>[]? Items { get; set; }

        [ProtoMember(4)]
        internal ConcurrentStack<int>? RemovedIndexes { get; set; }

        [ProtoMember(5)]
        internal int EntryPointId = -1;

        [ProtoMember(6)]
        internal int Capacity;

        [ProtoMember(7)]
        internal int Length;

        [ProtoMember(8)]
        internal int Count;

        internal TVector[]? ParsedItems
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

        internal GraphDataSnapshot(GraphData<TVector, TDistance> data)
        {
            Nodes = data.Nodes.Where(n => n is not null).ToArray();
            Items = data.Items.Where(i => i is not null).Select(i => new NestedArrayWrapper<TVector>(i)).ToArray();
            ActiveNodes = data.ActiveIds;
            RemovedIndexes = data.RemovedIndexes;
            EntryPointId = data.EntryPointId;
            Capacity = data.Capacity;
            Length = data.Length;
            Count = data.Count;
        }
    }
}