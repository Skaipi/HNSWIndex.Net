using ProtoBuf;

namespace HNSWIndex
{
    [ProtoContract]
    class Node
    {
        [ProtoMember(1)]
        public int Id;

        public object OutEdgesLock { get; } = new();

        public object InEdgesLock { get; } = new();

        public List<List<int>> OutEdges { get; set; } = new();

        public List<List<int>> InEdges { get; set; } = new();

        public int MaxLayer => OutEdges.Count - 1;

        // Trick to serialize lists of lists
        [ProtoMember(2, Name = nameof(OutEdges))]
        private List<IntListWrapper> OutEdgesSerialized
        {
            get => OutEdges.Select(l => new IntListWrapper(l)).ToList();
            set => OutEdges = (value ?? new List<IntListWrapper>()).Select(w => w.Values).ToList();
        }

        // Trick to serialize lists of lists
        [ProtoMember(3, Name = nameof(InEdges))]
        private List<IntListWrapper> InEdgesSerialized
        {
            get => InEdges.Select(l => new IntListWrapper(l)).ToList();
            set => InEdges = (value ?? new List<IntListWrapper>()).Select(w => w.Values).ToList();
        }

        [ProtoAfterDeserialization]
        private void AfterDeserialization()
        {
            for (int i = 0; i <= MaxLayer; i++)
            {
                OutEdges[i] ??= new List<int>();
                InEdges[i] ??= new List<int>();
            }
        }
    }

    [ProtoContract]
    struct IntListWrapper
    {
        public IntListWrapper(List<int> values) => Values = values;

        [ProtoMember(1)]
        public List<int> Values;
    }
}