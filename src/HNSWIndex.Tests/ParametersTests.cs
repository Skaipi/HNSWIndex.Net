namespace HNSWIndex.Tests
{
    [TestClass]
    public sealed class ParametersTests
    {
        private List<float[]>? vectors;

        [TestInitialize]
        public void TestInitialize()
        {
            vectors = Utils.RandomVectors(128, 1000);
        }

        [TestMethod]
        public void TestBruteForceHeuristic()
        {
            Assert.IsNotNull(vectors);

            var parameters = new HNSWParameters<float> { Heuristic = BruteForceHeuristic };
            var index = new HNSWIndex<float[], float>(Metrics.CosineMetric.Compute, parameters);

            for (int i = 0; i < vectors.Count; i++)
            {
                Utils.Normalize(vectors[i]);
                index.Add(vectors[i]);
            }

            var recall = Utils.Recall(index, vectors, vectors);
            Console.WriteLine(recall);
            Assert.IsTrue(recall > 0.90);
        }


        [TestMethod]
        public void TestParameterMinNN()
        {
            Assert.IsNotNull(vectors);

            var parameters = new HNSWParameters<float> { MinNN = 1 };
            var index = new HNSWIndex<float[], float>(Metrics.CosineMetric.Compute, parameters);

            for (int i = 0; i < vectors.Count; i++)
            {
                Utils.Normalize(vectors[i]);
                index.Add(vectors[i]);
            }

            var recall = Utils.Recall(index, vectors, vectors);
            Assert.IsTrue(recall > 0.70 && recall < 0.90);
        }

        [TestMethod]
        public void TestParameterMaxCandidates()
        {
            Assert.IsNotNull(vectors);

            var parameters = new HNSWParameters<float> { MaxCandidates = 32 };
            var index = new HNSWIndex<float[], float>(Metrics.CosineMetric.Compute, parameters);

            for (int i = 0; i < vectors.Count; i++)
            {
                Utils.Normalize(vectors[i]);
                index.Add(vectors[i]);
            }

            var recall = Utils.Recall(index, vectors, vectors);
            Assert.IsTrue(recall > 0.90);
        }

        [TestMethod]
        public void TestParameterLowRecall()
        {
            Assert.IsNotNull(vectors);

            var parameters = new HNSWParameters<float> { MaxEdges = 8, MinNN = 1, MaxCandidates = 16 };
            var index = new HNSWIndex<float[], float>(Metrics.CosineMetric.Compute, parameters);

            for (int i = 0; i < vectors.Count; i++)
            {
                Utils.Normalize(vectors[i]);
                index.Add(vectors[i]);
            }

            var recall = Utils.Recall(index, vectors, vectors);
            Assert.IsTrue(recall < 0.50);
        }

        [TestMethod]
        public void TestParameterAllowRemovals()
        {
            Assert.IsNotNull(vectors);

            var parameters = new HNSWParameters<float> { AllowRemovals = false };
            var index = new HNSWIndex<float[], float>(Metrics.SquaredEuclideanMetric.Compute, parameters);

            index.Add(vectors);

            var recall = Utils.Recall(index, vectors, vectors);
            Assert.IsTrue(recall > 0.9);

            var info = index.GetInfo();
            foreach (var layer in info.Layers)
            {
                Assert.IsTrue(layer.MaxInEdges == 0);
            }

            Assert.ThrowsException<InvalidOperationException>(() => index.Remove(0));
        }

        public static EdgeList BruteForceHeuristic(NodeDistance<float>[] candidates, Func<int, int, float> distanceFnc, int maxEdges)
        {
            return new EdgeList(candidates.OrderBy(x => x.Dist).Take(maxEdges).ToList().ConvertAll(x => x.Id));
        }
    }
}
