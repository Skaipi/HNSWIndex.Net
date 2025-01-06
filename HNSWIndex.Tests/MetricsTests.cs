namespace HNSWIndex.Tests
{
    [TestClass]
    public sealed class MetricsTests
    {
        [TestMethod]
        public void SquareEuclideanDistanceTest()
        {
            var a = Utils.RandomVectors(100, 1)[0];
            var b = Utils.RandomVectors(100, 1)[0];
            
            float epsilon = 1e-6f;
            var result = Metrics.SquaredEuclideanMetric.Compute(a, b);
            var reference = SquareEuclideanDistanceReference(a, b);

            Assert.IsTrue(Math.Abs(result - reference) < epsilon);
        }

        [TestMethod]
        public void CosineDistanceTest()
        {
            var a = Utils.RandomVectors(100, 1)[0];
            var b = Utils.RandomVectors(100, 1)[0];

            float epsilon = 1e-6f;
            var result = Metrics.CosineMetric.Compute(a, b);
            var reference = CosineDistanceReference(a, b);

            Assert.IsTrue(Math.Abs(result - reference) < epsilon);
        }

        [TestMethod]
        public void UnitCosineDistanceTest()
        {
            var a = Utils.RandomVectors(100, 1)[0];
            var b = Utils.RandomVectors(100, 1)[0];
            Utils.Normalize(a);
            Utils.Normalize(b);

            float epsilon = 1e-6f;
            var result = Metrics.CosineMetric.UnitCompute(a, b);
            var reference = UnitCosineDistanceReference(a, b);

            Assert.IsTrue(Math.Abs(result - reference) < epsilon);
        }

        public static float SquareEuclideanDistanceReference(float[] a, float[] b)
        {
            float result = 0;
            for (int i = 0; i < a.Length; i++)
            {
                var diff = a[i] - b[i];
                result += diff * diff;
            }
            return result;
        }

        public static float CosineDistanceReference(float[] a, float[] b)
        {
            float dot = 0f;
            float normA = 0f;
            float normB = 0f;
            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }

            float denom = (float)(Math.Sqrt(normA) * Math.Sqrt(normB));
            if (denom < 1e-30f)
                return 1f;
            return 1f - dot / denom;
        }

        public static float UnitCosineDistanceReference(float[] a, float[] b)
        {
            float result = 0;
            for (int i = 0; i < a.Length; i++)
            {
                result += a[i] * b[i];
            }
            return 1 - result;
        }
    }
}
