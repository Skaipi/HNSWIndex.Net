namespace HNSWIndex
{
    struct Node
    {
        public int Id;

        public object OutEdgesLock;

        public object InEdgesLock;

        public List<List<int>> OutEdges;

        public List<List<int>> InEdges;

        public int MaxLayer => OutEdges.Count - 1;
    }
}