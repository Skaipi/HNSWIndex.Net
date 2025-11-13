namespace HNSWIndex.Tests
{
    using HNSWIndex;

    [TestClass]
    public sealed class GraphSerializationTests
    {
        private List<float[]>? vectors;

        [TestInitialize]
        public void TestInitialize()
        {
            vectors = Utils.RandomVectors(128, 2_000);
        }

        [TestMethod]
        public void EncodeDecodeTest()
        {
            Assert.IsNotNull(vectors);

            var path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
            var index = new HNSWIndex<float[], float>(Metrics.SquaredEuclideanMetric.Compute);

            for (int i = 0; i < vectors.Count; i++)
                index.Add(vectors[i]);

            try
            {
                index.Serialize(path);
                var decodedIndex = HNSWIndex<float[], float>.Deserialize(Metrics.SquaredEuclideanMetric.Compute, path);

                for (int i = 0; i < vectors.Count; i++)
                {
                    var originalResults = index.KnnQuery(vectors[i], 5);
                    var decodeResults = decodedIndex.KnnQuery(vectors[i], 5);
                    for (int j = 0; j < originalResults.Count; j++)
                    {
                        Assert.AreEqual(originalResults[j].Id, decodeResults[j].Id);
                        Assert.IsTrue(originalResults[j].Label.SequenceEqual(decodeResults[j].Label));
                        Assert.AreEqual(originalResults[j].Distance, decodeResults[j].Distance);
                    }
                }
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
        }
    }
}