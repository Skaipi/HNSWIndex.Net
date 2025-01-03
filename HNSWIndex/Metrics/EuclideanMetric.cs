using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;

namespace HNSWIndex.Metrics
{
    public class SquaredEuclideanMetric
    {
        // NOTE: We assume a and b have the same dimension
        public static unsafe float Compute(float[] a, float[] b)
        {
            int i = 0;
            int length = a.Length;

            fixed (float* pa = a, pb = b)
            {
                float* ptrA = pa;
                float* ptrB = pb;

                if (Avx.IsSupported)
                {
                    Vector256<float> accumulator256 = Vector256<float>.Zero;
                    int step = Vector256<float>.Count;
                    int stop = length - step;

                    for (; i <= stop; i += step)
                    {
                        Vector256<float> vu = Avx.LoadVector256(ptrA + i);
                        Vector256<float> vv = Avx.LoadVector256(ptrB + i);
                        Vector256<float> diff = Avx.Subtract(vu, vv);
                        Vector256<float> dist = Avx.Multiply(diff, diff);
                        accumulator256 = Avx.Add(accumulator256, dist);
                    }

                    Vector256<float> temp = Avx.HorizontalAdd(accumulator256, accumulator256);
                    temp = Avx.HorizontalAdd(temp, temp);
                    float partialSum = temp.GetElement(0) + temp.GetElement(1);

                    // Handle remainder
                    for (; i < length; i++)
                    {
                        float diff = ptrA[i] - ptrB[i];
                        partialSum += diff * diff;
                    }

                    return partialSum;
                }
                else if (Sse.IsSupported)
                {
                    Vector128<float> accumulator = Vector128<float>.Zero;
                    int step = Vector128<float>.Count;
                    int stop = length - step;
                    for (; i <= stop; i += step)
                    {
                        Vector128<float> vu = Sse.LoadVector128(ptrA + i);
                        Vector128<float> vv = Sse.LoadVector128(ptrB + i);
                        Vector128<float> diff = Sse.Subtract(vu, vv);
                        Vector128<float> dist = Sse.Multiply(diff, diff);
                        accumulator = Sse.Add(accumulator, dist);
                    }

                    Vector128<float> sumPair = Sse.Add(accumulator, Sse.MoveHighToLow(accumulator, accumulator));
                    Vector128<float> finalSum = Sse.Add(sumPair, Sse.Shuffle(sumPair, sumPair, 0x55));
                    float partialSum = Sse41.IsSupported
                        ? Sse41.Extract(finalSum, 0)
                        : Unsafe.ReadUnaligned<float>(&finalSum);

                    for (; i < length; i++)
                    {
                        float diff = ptrA[i] - ptrB[i];
                        partialSum += diff * diff;
                    }

                    return partialSum;
                }
                else
                {
                    float sum = 0f;
                    for (; i < length; i++)
                    {
                        float diff = ptrA[i] - ptrB[i];
                        sum += diff * diff;
                    }
                    return sum;
                }
            }
        }
    }
}
