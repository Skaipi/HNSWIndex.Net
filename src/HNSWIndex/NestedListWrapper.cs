using System.Collections;
using ProtoBuf;

namespace HNSWIndex
{
    [ProtoContract]
    internal struct NestedListWrapper<T>
    {
        public NestedListWrapper(List<T> values) => Values = values;

        [ProtoMember(1)]
        public List<T> Values;
    }

    [ProtoContract]
    internal struct NestedArrayWrapper<T> where T : IList
    {
        public NestedArrayWrapper(T values) => Values = values;

        [ProtoMember(1)]
        public T Values;
    }
}