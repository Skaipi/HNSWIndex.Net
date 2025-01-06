using System.Numerics;
using System.Runtime.CompilerServices;

namespace HNSWIndex
{
    public class HNSWIndex<TItem, TDistance> where TDistance : struct, IFloatingPoint<TDistance>
    {
        private Func<TItem, TItem, TDistance> distanceFnc;

        private readonly HNSWParameters<TDistance> parameters;

        private readonly GraphData<TItem, TDistance> data;

        private readonly GraphConnector<TItem, TDistance> connector;

        private readonly GraphNavigator<TItem, TDistance> navigator;

        public HNSWIndex(Func<TItem, TItem, TDistance> distFnc, HNSWParameters<TDistance>? hnswParameters = null)
        {
            hnswParameters ??= new HNSWParameters<TDistance>();
            distanceFnc = distFnc;
            parameters = hnswParameters;

            data = new GraphData<TItem, TDistance>(distFnc, hnswParameters);
            navigator = new GraphNavigator<TItem, TDistance>(data);
            connector = new GraphConnector<TItem, TDistance>(data, navigator, hnswParameters);

            data.Reallocated += OnDataResized;
        }

        public void Add(TItem item)
        {
            var itemId = -1;
            lock (data.indexLock)
            {
                itemId = data.AddItem(item);
            }

            lock (data.Nodes[itemId].OutEdgesLock)
            {
                connector.ConnectNewNode(itemId);
            }
        }

        public void Remove(int itemIndex)
        {
            if (itemIndex == data.EntryPointId)
            {
                data.RemoveEntryPoint();
            }

            connector.RemoveConnections(itemIndex);
            data.RemoveItem(itemIndex);
        }

        public List<KNNResult<TItem, TDistance>> KnnQuery(TItem query, int k, Func<TItem, bool>? filterFnc = null)
        {
            if (data.Nodes.Count == 0) return new List<KNNResult<TItem, TDistance>>();

            Func<int, bool> indexFilter = _ => true;
            if (filterFnc is not null)
                indexFilter = (index) => filterFnc(data.Items[index]);


            TDistance queryDistance(int nodeId, TItem label)
            {
                return distanceFnc(data.Items[nodeId], label);
            }

            var neighboursAmount = Math.Max(parameters.MinNN, k);
            var distCalculator = new DistanceCalculator<TItem, TDistance>(queryDistance, query);
            var ep = navigator.FindEntryPoint(0, distCalculator);
            var topCandidates = navigator.SearchLayer(ep.Id, 0, neighboursAmount, distCalculator, indexFilter);

            if (k < neighboursAmount)
            {
                return topCandidates.OrderBy(c => c.Dist).Take(k).ToList().ConvertAll(c => new KNNResult<TItem, TDistance>(c.Id, data.Items[c.Id], c.Dist));
            }
            return topCandidates.ConvertAll(c => new KNNResult<TItem, TDistance>(c.Id, data.Items[c.Id], c.Dist));
        }

        public HNSWInfo GetInfo()
        {
            return new HNSWInfo(data.Nodes, data.GetTopLayer());
        }

        private void OnDataResized(object? sender, ReallocateEventArgs e)
        {
            navigator.OnReallocate(e.NewCapacity);
        }
    }
}
