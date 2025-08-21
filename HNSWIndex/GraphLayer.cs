namespace HNSWIndex
{
    /// <summary>
    /// Public API to graph structure of HNSW
    /// </summary>
    public class GraphLayer
    {
        public int Layer;
        public List<VertexHNSW> Vertices = new();

        internal GraphLayer(Node[] nodes, int layer)
        {
            Layer = layer;
            foreach (var node in nodes)
            {
                Vertices.Add(new VertexHNSW(node, layer));
            }
        }
    }

    /// <summary>
    /// Public API to important Node data
    /// </summary>
    public class VertexHNSW
    {
        public readonly int Id;

        public readonly List<int> Neighbours;

        internal VertexHNSW(Node node, int layer)
        {
            Id = node.Id;
            Neighbours = node.OutEdges[layer];
        }
    }
}
