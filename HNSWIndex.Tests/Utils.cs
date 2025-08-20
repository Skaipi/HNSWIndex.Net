using System.Numerics;

namespace HNSWIndex.Tests
{
    internal static class Utils
    {
        private static int seed = 65537;

        internal static float Magnitude(float[] vector)
        {
            float magnitude = 0.0f;
            int step = Vector<float>.Count;
            for (int i = 0; i < vector.Length; i++)
            {
                magnitude += vector[i] * vector[i];
            }
            return (float)Math.Sqrt(magnitude);
        }

        internal static void Normalize(float[] vector)
        {
            float normFactor = 1f / Magnitude(vector);
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] *= normFactor;
            }
        }

        internal static List<float[]> RandomVectors(int vectorSize, int vectorsCount)
        {
            var random = new Random(seed);
            var vectors = new List<float[]>();

            for (int i = 0; i < vectorsCount; i++)
            {
                var vector = new float[vectorSize];
                for (int d = 0; d < vectorSize; d++)
                    vector[d] = random.NextSingle();
                vectors.Add(vector);
            }

            return vectors;
        }

        /// <summary>
        /// Compute Recall@k - is groundtruth in first k nearest neighbors
        /// </summary>
        internal static float Recall(HNSWIndex<float[], float> index, List<float[]> vectors, List<float[]> groundTrouths, int k = 1)
        {
            if (vectors.Count != groundTrouths.Count) throw new ArgumentException("Queries and Groundtrouths size mismatch");

            var goodFinds = 0;
            for (int i = 0; i < vectors.Count; i++)
            {
                var result = index.KnnQuery(vectors[i], k);
                for (int j = 0; j < k; j++)
                {
                    var candidate = result[j].Label;
                    if (candidate.SequenceEqual(groundTrouths[i]))
                        goodFinds++;
                }
            }
            return (float)goodFinds / vectors.Count;
        }
    }
}
