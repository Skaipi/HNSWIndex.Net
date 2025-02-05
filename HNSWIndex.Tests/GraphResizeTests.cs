namespace HNSWIndex.Tests
{
    using HNSWIndex;

    [TestClass]
    public class GraphResizeTests
    {
        private List<float[]>? vectors;

        [TestInitialize]
        public void TestInitialize()
        {
            vectors = Utils.RandomVectors(128, 5000);
        }

        [TestMethod]
        public void SingleThreadGraphResize()
        {
            Assert.IsNotNull(vectors);

            var parameters = new HNSWParameters<float>() { CollectionSize = 10 };
            var index = new HNSWIndex<float[], float>(Metrics.SquaredEuclideanMetric.Compute);

            for (int i = 0; i < vectors.Count; i++)
            {
                Utils.Normalize(vectors[i]);
                index.Add(vectors[i]);
            }

            var goodFinds = 0;
            for (int i = 0; i < vectors.Count; i++)
            {
                var result = index.KnnQuery(vectors[i], 1);
                var bestFound = result[0].Label;
                if (vectors[i] == bestFound)
                    goodFinds++;
            }

            var recall = (float)goodFinds / vectors.Count;
            Assert.IsTrue(recall > 0.85);

            // Ensure in and out edges are balanced
            var info = index.GetInfo();
            foreach (var layer in info.Layers)
            {
                Assert.IsTrue(layer.AvgOutEdges == layer.AvgInEdges);
            }
        }

        [TestMethod]
        public void MultiThreadGraphResize()
        {
            Assert.IsNotNull(vectors);

            var parameters = new HNSWParameters<float>() { CollectionSize = 10 };
            var index = new HNSWIndex<float[], float>(Metrics.SquaredEuclideanMetric.Compute);

            Parallel.For(0, vectors.Count, i =>
            {
                Utils.Normalize(vectors[i]);
                index.Add(vectors[i]);
            });

            var goodFinds = 0;
            for (int i = 0; i < vectors.Count; i++)
            {
                var result = index.KnnQuery(vectors[i], 1);
                var bestFound = result[0].Label;
                if (vectors[i] == bestFound)
                    goodFinds++;
            }

            var recall = (float)goodFinds / vectors.Count;
            Assert.IsTrue(recall > 0.85);

            // Ensure in and out edges are balanced
            var info = index.GetInfo();
            foreach (var layer in info.Layers)
            {
                Assert.IsTrue(layer.AvgOutEdges == layer.AvgInEdges);
            }
        }
    }
}
