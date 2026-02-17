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

        [TestMethod]
        public void ActiveSetContainsCorrectTest()
        {
            Assert.IsNotNull(vectors);

            var index = new HNSWIndex<float[], float>(Metrics.SquaredEuclideanMetric.Compute);
            var evenIndexedVectors = new List<(float[] Label, int Id)>();
            var oddIndexedVectors = new List<(float[] Label, int Id)>();
            for (int i = 0; i < vectors.Count; i++)
            {
                Utils.Normalize(vectors[i]);
                var id = index.Add(vectors[i]);
                if (i % 2 == 0) evenIndexedVectors.Add((vectors[i], id));
                else oddIndexedVectors.Add((vectors[i], id));
            }

            for (int i = 0; i < oddIndexedVectors.Count; i++)
            {
                index.Remove(oddIndexedVectors[i].Id);
            }

            Assert.IsTrue(index.Count == evenIndexedVectors.Count);

            foreach (var id in index.Ids())
            {
                Assert.IsTrue(evenIndexedVectors.Any(v => v.Id == id));
                Assert.IsFalse(oddIndexedVectors.Any(v => v.Id == id));
            }
        }

        [TestMethod]
        public void RemoveAllTest()
        {
            Assert.IsNotNull(vectors);

            var parameters = new HNSWParameters<float>() { CollectionSize = 10 };
            var index = new HNSWIndex<float[], float>(Metrics.SquaredEuclideanMetric.Compute, parameters);
            for (int i = 0; i < vectors.Count; i++) index.Add(vectors[i]);
            for (int i = 0; i < vectors.Count; i++)
            {
                index.Remove(i);
                Assert.IsTrue(index.Count == vectors.Count - i - 1);
            }
        }

        [TestMethod]
        public void RemoveAllParallelTest()
        {
            Assert.IsNotNull(vectors);

            var parameters = new HNSWParameters<float>() { CollectionSize = 10 };
            var index = new HNSWIndex<float[], float>(Metrics.SquaredEuclideanMetric.Compute, parameters);
            index.Add(vectors);
            Parallel.For(0, vectors.Count, i =>
            {
                index.Remove(i);
            });

            Assert.IsTrue(index.Count == 0);
        }
    }
}
