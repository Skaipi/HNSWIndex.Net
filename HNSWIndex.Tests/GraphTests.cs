namespace HNSWIndex.Tests
{
    using HNSWIndex;
    using System.Numerics;

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

            var index = new HNSWIndex<float[], float>(Metrics.CosineMetric.Compute);
            for (int i = 0; i < vectors.Count; i++)
            {
                Utils.Normalize(vectors[i]);
                index.Add(vectors[i]);
            }

            var goodFinds = 0;
            for (int i=0; i< vectors.Count; i++)
            {
                var result = index.KnnQuery(vectors[i], 1);
                var bestItem = result[0].Label;
                if (vectors[i] == bestItem)
                    goodFinds++;
            }

            var recall = (float)goodFinds / vectors.Count;
            Assert.IsTrue(recall > 0.70);
        }

        [TestMethod]
        public void BuildGraphMultiThread()
        {
            Assert.IsNotNull(vectors);

            var index = new HNSWIndex<float[], float>(Metrics.CosineMetric.Compute);
            Parallel.For(0, vectors.Count, i => {
                Utils.Normalize(vectors[i]);
                index.Add(vectors[i]);
            });

            var goodFinds = 0;
            for (int i = 0; i < vectors.Count; i++)
            {
                var result = index.KnnQuery(vectors[i], 1);
                var bestItem = result[0].Label;
                if (vectors[i] == bestItem)
                    goodFinds++;
            }

            var recall = (float)goodFinds / vectors.Count;
            Assert.IsTrue(recall > 0.70);
        }
    }
}
