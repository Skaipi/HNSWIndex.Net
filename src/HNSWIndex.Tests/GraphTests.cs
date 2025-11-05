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
            vectors = Utils.RandomVectors(128, 2_000);
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
        public void BuildGraphMultiThread()
        {
            Assert.IsNotNull(vectors);

            var index = new HNSWIndex<float[], float>(Metrics.CosineMetric.UnitCompute);
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
        public void BuildGraphBatch()
        {
            Assert.IsNotNull(vectors);

            // NOTE: We omit normalization step in this test
            var index = new HNSWIndex<float[], float>(Metrics.CosineMetric.Compute);
            index.Add(vectors);

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
            for (int i = 0; i < vectors.Count; i++)
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

            for (int i = 0; i < vectors.Count; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    Assert.IsTrue(singleThreadResults[i][j].Id == multiThreadResults[i][j].Id);
                }
            }
        }

        [TestMethod]
        public void RemoveNodesTest()
        {
            Assert.IsNotNull(vectors);

            var index = new HNSWIndex<float[], float>(Metrics.CosineMetric.UnitCompute);
            var evenIndexedVectors = new List<(float[] Label, int Id)>();
            var oddIndexedVectors = new List<(float[] Label, int Id)>();
            for (int i = 0; i < vectors.Count; i++)
            {
                Utils.Normalize(vectors[i]);
                var id = index.Add(vectors[i]);
                if (i % 2 == 0) evenIndexedVectors.Add((vectors[i], id));
                else oddIndexedVectors.Add((vectors[i], id));
            }

            var insertRecall = Utils.Recall(index, vectors, vectors);

            for (int i = 0; i < oddIndexedVectors.Count; i++)
            {
                index.Remove(oddIndexedVectors[i].Id);
            }

            var evenVectors = evenIndexedVectors.ConvertAll(v => v.Label);
            var removalRecall = Utils.Recall(index, evenVectors, evenVectors);

            // Allow 10% drop after removal
            Assert.IsTrue(insertRecall < removalRecall + 0.1 * insertRecall);

            // Ensure in and out edges are balanced
            var info = index.GetInfo();
            foreach (var layer in info.Layers)
            {
                Assert.IsTrue(layer.AvgOutEdges == layer.AvgInEdges);
            }
        }

        [TestMethod]
        public void RemoveNodesParallelTest()
        {
            Assert.IsNotNull(vectors);

            var index = new HNSWIndex<float[], float>(Metrics.CosineMetric.UnitCompute);
            var evenIndexedVectors = new List<(float[] Label, int Id)>();
            var oddIndexedVectors = new List<(float[] Label, int Id)>();
            for (int i = 0; i < vectors.Count; i++)
            {
                Utils.Normalize(vectors[i]);
                var id = index.Add(vectors[i]);
                if (i % 2 == 0) evenIndexedVectors.Add((vectors[i], id));
                else oddIndexedVectors.Add((vectors[i], id));
            }

            var insertRecall = Utils.Recall(index, vectors, vectors);

            Parallel.For(0, oddIndexedVectors.Count, (i) =>
            {
                index.Remove(oddIndexedVectors[i].Id);
            });

            var evenVectors = evenIndexedVectors.ConvertAll(v => v.Label);
            var removalRecall = Utils.Recall(index, evenVectors, evenVectors);

            // Allow 10% drop after removal
            Assert.IsTrue(insertRecall < removalRecall + 0.1 * insertRecall);

            // Ensure in and out edges are balanced
            var info = index.GetInfo();
            foreach (var layer in info.Layers)
            {
                Assert.IsTrue(layer.AvgOutEdges == layer.AvgInEdges);
            }
        }

        [TestMethod]
        public void RemoveNodesBatchTest()
        {
            Assert.IsNotNull(vectors);

            var index = new HNSWIndex<float[], float>(Metrics.CosineMetric.UnitCompute);
            var evenIndexedVectors = new List<(float[] Label, int Id)>();
            var oddIndexedVectors = new List<(float[] Label, int Id)>();
            for (int i = 0; i < vectors.Count; i++)
            {
                Utils.Normalize(vectors[i]);
                var id = index.Add(vectors[i]);
                if (i % 2 == 0) evenIndexedVectors.Add((vectors[i], id));
                else oddIndexedVectors.Add((vectors[i], id));
            }


            var insertRecall = Utils.Recall(index, vectors, vectors);
            index.Remove(oddIndexedVectors.ConvertAll(x => x.Id));

            var evenVectors = evenIndexedVectors.ConvertAll(v => v.Label);
            var removalRecall = Utils.Recall(index, evenVectors, evenVectors);

            // Allow 10% drop after removal
            Assert.IsTrue(insertRecall < removalRecall + 0.1 * insertRecall);

            // Ensure in and out edges are balanced
            var info = index.GetInfo();
            foreach (var layer in info.Layers)
            {
                Assert.IsTrue(layer.AvgOutEdges == layer.AvgInEdges);
            }
        }

        [TestMethod]
        public void UpdateNodesTest()
        {
            Assert.IsNotNull(vectors);

            var dim = vectors[0].Length;
            var newVectors = Utils.RandomVectors(dim, vectors.Count);
            var indexes = new List<int>(vectors.Count);
            var index = new HNSWIndex<float[], float>(Metrics.SquaredEuclideanMetric.Compute);
            for (int i = 0; i < vectors.Count; i++)
            {
                Utils.Normalize(vectors[i]);
                indexes.Add(index.Add(vectors[i]));
            }

            var recall = Utils.Recall(index, vectors, vectors);

            index.Update(indexes, newVectors);

            var updateRecall = Utils.Recall(index, newVectors, newVectors);
            Assert.IsTrue(recall < updateRecall + 0.05 * recall);

            // Ensure in and out edges are balanced
            // var info = index.GetInfo();
            // foreach (var layer in info.Layers)
            // {
            //     Assert.IsTrue(layer.AvgOutEdges == layer.AvgInEdges);
            // }
        }
    }
}
