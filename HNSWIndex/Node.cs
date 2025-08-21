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
        private List<NestedListWrapper<int>> OutEdgesSerialized
        {
            get => OutEdges.Select(l => new NestedListWrapper<int>(l)).ToList();
            set => OutEdges = (value ?? new List<NestedListWrapper<int>>()).Select(w => w.Values).ToList();
        }

        // Trick to serialize lists of lists
        [ProtoMember(3, Name = nameof(InEdges))]
        private List<NestedListWrapper<int>> InEdgesSerialized
        {
            get => InEdges.Select(l => new NestedListWrapper<int>(l)).ToList();
            set => InEdges = (value ?? new List<NestedListWrapper<int>>()).Select(w => w.Values).ToList();
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
}