using System.Collections.Concurrent;

namespace HNSWIndex
{
    public class HNSWInfo
    {
        public List<LayerInfo> Layers;

        internal HNSWInfo(Node[] nodes, ConcurrentQueue<int> removedNodes, int maxLayer, bool allowRemovals)
        {
            Layers = new List<LayerInfo>(maxLayer + 1);
            for (int layer = 0; layer <= maxLayer; layer++)
            {
                Layers.Add(new LayerInfo(nodes.Where(x => x is not null && x.MaxLayer >= layer && !removedNodes.Contains(x.Id)).ToList(), layer, allowRemovals));
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
            public int OutEdgesMedian;
            public int InEdgesMedian;

            internal LayerInfo(List<Node> nodesOnLayer, int layer, bool hasInEdges)
            {
                LayerId = layer;
                NodesCount = nodesOnLayer.Count;
                MaxOutEdges = nodesOnLayer.Max(x => x.OutEdges[layer].Count);
                MinOutEdges = nodesOnLayer.Min(x => x.OutEdges[layer].Count);
                AvgOutEdges = nodesOnLayer.Average(x => x.OutEdges[layer].Count);
                OutEdgesMedian = Median(nodesOnLayer.ConvertAll(x => x.OutEdges[layer].Count));
                MaxInEdges = hasInEdges ? nodesOnLayer.Max(x => x.InEdges[layer].Count) : 0;
                MinInEdges = hasInEdges ? nodesOnLayer.Min(x => x.InEdges[layer].Count) : 0;
                AvgInEdges = hasInEdges ? nodesOnLayer.Average(x => x.InEdges[layer].Count) : 0;
                InEdgesMedian = hasInEdges ? Median(nodesOnLayer.ConvertAll(x => x.InEdges[layer].Count)) : 0;
            }

            private int Median(List<int> arr)
            {
                var sorted = arr.OrderBy(x => x).ToList();
                int count = sorted.Count;
                if (count % 2 == 0) return (sorted[count / 2 - 1] + sorted[count / 2]) / 2;
                else return sorted[count / 2];
            }
        }
    }
}
