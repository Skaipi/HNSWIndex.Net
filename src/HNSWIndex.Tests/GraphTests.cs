namespace HNSWIndex.Tests
{
    using System.Runtime.CompilerServices;
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

            Assert.IsTrue(insertRecall * 0.98 < removalRecall);

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

            Assert.IsTrue(insertRecall * 0.98 < removalRecall);

            // Ensure in and out edges are balanced
            var info = index.GetInfo();
            foreach (var layer in info.Layers)
            {
                Assert.IsTrue(layer.AvgOutEdges == layer.AvgInEdges);
            }
        }

        [TestMethod]
        public void RemoveAndReleaseNodesBatchTest()
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

            Assert.IsTrue(insertRecall * 0.98 < removalRecall);

            // Ensure in and out edges are balanced after remove
            var removeInfo = index.GetInfo();
            foreach (var layer in removeInfo.Layers)
            {
                Assert.IsTrue(layer.AvgOutEdges == layer.AvgInEdges);
            }

            index.ReleaseItems(oddIndexedVectors.ConvertAll(x => x.Id));
            var releaseRecall = Utils.Recall(index, evenVectors, evenVectors);

            Assert.IsTrue(releaseRecall == removalRecall);
            foreach (var (_, id) in oddIndexedVectors)
            {
                Assert.IsNull(index.Data.Items[id]);
                Assert.IsNull(index.Data.Nodes[id]);
            }

            // Ensure in and out edges are balanced after release
            var releaseInfo = index.GetInfo();
            for (int i = 0; i < removeInfo.Layers.Count; i++)
            {
                var layer = releaseInfo.Layers[i];
                Assert.IsTrue(layer.AvgOutEdges == layer.AvgInEdges);
                Assert.IsTrue(layer.AvgOutEdges == removeInfo.Layers[i].AvgOutEdges);
            }
        }


        [MethodImpl(MethodImplOptions.NoInlining)]
        private static (int Id, WeakReference<Utils.TrackedVector> Reference) AddAndRemove(
            HNSWIndex<Utils.TrackedVector, float> index)
        {
            var item = new Utils.TrackedVector { Value = 1 };
            var reference = new WeakReference<Utils.TrackedVector>(item);
            var id = index.Add(item);
            index.Remove(id);

            return (id, reference);
        }

        private static void ForceFullCollection()
        {
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
            GC.WaitForPendingFinalizers();
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static bool IsAlive<T>(WeakReference<T> reference)
            where T : class
        {
            return reference.TryGetTarget(out _);
        }

        [TestMethod]
        [DoNotParallelize]
        public void ReleaseItemAllowsRemovedItemToBeCollected()
        {
            var index = new HNSWIndex<Utils.TrackedVector, float>(
                static (a, b) => Math.Abs(a.Value - b.Value));

            var (id, reference) = AddAndRemove(index);

            ForceFullCollection();
            // Strong reference still present
            Assert.IsTrue(IsAlive(reference));

            index.ReleaseItem(id);

            Assert.IsNull(index.Data.Items[id]);
            Assert.IsNull(index.Data.Nodes[id]);

            ForceFullCollection();
            // No strong references left
            Assert.IsFalse(IsAlive(reference));

            var replacement = new Utils.TrackedVector { Value = 2 };
            var replacementId = index.Add(replacement);

            Assert.AreEqual(id, replacementId);
            Assert.AreSame(replacement, index.Data.Items[id]);
            Assert.IsNotNull(index.Data.Nodes[id]);
            Assert.AreEqual(1, index.Count);
        }

        [TestMethod]
        public void ReleaseActiveItemThrows()
        {
            var item = new Utils.TrackedVector { Value = 1 };
            var index = new HNSWIndex<Utils.TrackedVector, float>(
                static (a, b) => Math.Abs(a.Value - b.Value));
            var id = index.Add(item);

            Assert.ThrowsException<InvalidOperationException>(() => index.ReleaseItem(id));
            Assert.AreEqual(1, index.Count);
            Assert.AreSame(item, index.Items().Single());
            Assert.AreSame(item, index.Data.Items[id]);
            Assert.IsNotNull(index.Data.Nodes[id]);
        }

        [TestMethod]
        public void RangeQueryTest()
        {
            Assert.IsNotNull(vectors);

            var index = new HNSWIndex<float[], float>(Metrics.SquaredEuclideanMetric.Compute);
            for (int i = 0; i < vectors.Count; i++)
            {
                index.Add(vectors[i]);
            }

            var range = 32.0f;

            // ensure all results are within the range
            var batchResults = index.BatchRangeQuery(vectors, range);
            foreach (var results in batchResults)
                Assert.IsTrue(results.All(r => r.Distance <= range));
        }

        [TestMethod]
        public void ConnectedComponentCountsEmptyGraphTest()
        {
            var index = new HNSWIndex<float[], float>(Metrics.CosineMetric.UnitCompute);
            CollectionAssert.AreEqual(Array.Empty<int>(), index.GetConnectedComponentCounts());
        }

        [TestMethod]
        public void ConnectedComponentCountsPerLayerTest()
        {
            Assert.IsNotNull(vectors);

            var parameters = new HNSWParameters<float> { RandomSeed = 12345 };
            var index = new HNSWIndex<float[], float>(Metrics.CosineMetric.UnitCompute, parameters);

            for (int i = 0; i < 256; i++)
            {
                Utils.Normalize(vectors[i]);
                index.Add(vectors[i]);
            }

            var counts = index.GetConnectedComponentCounts();
            Assert.IsTrue(counts.Length >= 1);
            foreach (var count in counts)
            {
                Assert.AreEqual(1, count);
            }
        }
    }
}
