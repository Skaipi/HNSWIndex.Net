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
            var index = new HNSWIndex<float[], float>(Metrics.SquaredEuclideanMetric.Compute, parameters);

            for (int i = 0; i < vectors.Count; i++)
            {
                Utils.Normalize(vectors[i]);
                index.Add(vectors[i]);
            }

            var recall = Utils.Recall(index, vectors, vectors);
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
            var index = new HNSWIndex<float[], float>(Metrics.SquaredEuclideanMetric.Compute, parameters);

            Parallel.For(0, vectors.Count, i =>
            {
                Utils.Normalize(vectors[i]);
                index.Add(vectors[i]);
            });

            var recall = Utils.Recall(index, vectors, vectors);
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
