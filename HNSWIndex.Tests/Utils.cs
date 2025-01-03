using System.Numerics;

namespace HNSWIndex.Tests
{
    internal static class Utils
    {
        internal static float Magnitude(float[] vector)
        {
            float magnitude = 0.0f;
            int step = Vector<float>.Count;
            for (int i=0; i < vector.Length; i++)
            {
                magnitude += vector[i] * vector[i];
            }
            return (float)Math.Sqrt(magnitude);
        }

        internal static void Normalize(float[] vector)
        {
            float normFactor = 1f / Magnitude(vector);
            for (int i=0; i < vector.Length; i++)
            {
                vector[i] *= normFactor;
            }
        }

        internal static List<float[]> RandomVectors(int vectorSize, int vectorsCount)
        {
            var random = new Random(vectorsCount);
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
    }
}
