namespace HNSWIndex.Tests
{
    using HNSWIndex;

    [TestClass]
    public sealed class GraphTests
    {
        private List<float[]>? vectors;

        [TestInitialize]
        public void TestInitialize()
        {
            vectors = Utils.RandomVectors(128, 5000);
        }

        [TestMethod]
        public void BuildGraphSingleThread()
        {
            Assert.IsNotNull(vectors);

            var index = new HNSWIndex<float[], float>(Metrics.CosineMetric.UnitCompute);
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
        public void BuildGraphMultiThread()
        {
            Assert.IsNotNull(vectors);

            var index = new HNSWIndex<float[], float>(Metrics.CosineMetric.UnitCompute);
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

        [TestMethod]
        public void QueryGraphMultiThread()
        {
            Assert.IsNotNull(vectors);

            var k = 10;
            var index = new HNSWIndex<float[], float>(Metrics.CosineMetric.UnitCompute);
            for (int i = 0; i < vectors.Count; i++)
            {
                Utils.Normalize(vectors[i]);
                index.Add(vectors[i]);
            }

            var singleThreadResults = new List<List<KNNResult<float[], float>>>(vectors.Count);
            var multiThreadResults = new List<List<KNNResult<float[], float>>>(vectors.Count);
            for (int i=0; i< vectors.Count; i++)
            {
                singleThreadResults.Add(new List<KNNResult<float[], float>>());
                multiThreadResults.Add(new List<KNNResult<float[], float>>());
            }

            for (int i = 0; i < vectors.Count; i++)
            {
                singleThreadResults[i] = index.KnnQuery(vectors[i], k);
            }

            Parallel.For(0, vectors.Count, i =>
            {
                multiThreadResults[i] = index.KnnQuery(vectors[i], k);
            });

            for (int i=0; i<vectors.Count; i++)
            {
                for (int j=0; j<k; j++)
                {
                    Assert.IsTrue(singleThreadResults[i][j].Id == multiThreadResults[i][j].Id);
                }
            }
        }
    }
}
