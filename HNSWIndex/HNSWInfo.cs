namespace HNSWIndex
{
    public class HNSWInfo
    {
        public List<LayerInfo> Layers;

        internal HNSWInfo(List<Node> nodes, HashSet<int> removedNodes, int maxLayer)
        {
            Layers = new List<LayerInfo>(maxLayer + 1);
            for (int layer = 0; layer <= maxLayer; layer++)
            {
                Layers.Add(new LayerInfo(nodes.Where(x => x.MaxLayer >= layer && !removedNodes.Contains(x.Id)).ToList(), layer));
            }
        }

        public class LayerInfo
        {
            public int LayerId;
            public int NodesCount;
            public int MaxOutEdges;
            public int MinOutEdges;
            public int MaxInEdges;
            public int MinInEdges;
            public double AvgOutEdges;
            public double AvgInEdges;

            internal LayerInfo(List<Node> nodesOnLayer, int layer)
            {
                LayerId = layer;
                NodesCount = nodesOnLayer.Count;
                MaxOutEdges = nodesOnLayer.Max(x => x.OutEdges[layer].Count);
                MinOutEdges = nodesOnLayer.Min(x => x.OutEdges[layer].Count);
                MaxInEdges = nodesOnLayer.Max(x => x.InEdges[layer].Count);
                MinInEdges = nodesOnLayer.Min(x => x.InEdges[layer].Count);
                AvgOutEdges = nodesOnLayer.Average(x => x.OutEdges[layer].Count);
                AvgInEdges = nodesOnLayer.Average(x => x.InEdges[layer].Count);
            }
        }
    }
}
