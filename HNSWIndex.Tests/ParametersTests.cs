﻿using System.Reflection.Metadata.Ecma335;

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

            var goodFinds = 0;
            for (int i = 0; i < vectors.Count; i++)
            {
                var result = index.KnnQuery(vectors[i], 1);
                var bestItem = result[0].Label;
                if (vectors[i] == bestItem)
                    goodFinds++;
            }

            var recall = (float)goodFinds / vectors.Count;
            Assert.IsTrue(recall > 0.90);
        }

        public static List<int> BruteForceHeuristic(List<NodeDistance<float>> candidates, Func<int, int, float> distanceFnc, int maxEdges)
        {
            return candidates.OrderBy(x => -x.Dist).Take(maxEdges).ToList().ConvertAll(x => x.Id);
        }
    }
}