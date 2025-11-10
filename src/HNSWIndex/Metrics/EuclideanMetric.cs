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

            fixed (float* ptrA = a, ptrB = b)
            {

                if (Avx.IsSupported)
                {
                    Vector256<float> acc256 = Vector256<float>.Zero;
                    int step = Vector256<float>.Count;
                    int stop = length & ~(step - 1);

                    for (; i + 2 * step <= stop; i += 2 * step)
                    {
                        var a0 = Avx.LoadVector256(ptrA + i);
                        var b0 = Avx.LoadVector256(ptrB + i);
                        var d0 = Avx.Subtract(a0, b0);
                        acc256 = Fma.IsSupported ? Fma.MultiplyAdd(d0, d0, acc256) : Avx.Add(acc256, Avx.Multiply(d0, d0));

                        var a1 = Avx.LoadVector256(ptrA + i + step);
                        var b1 = Avx.LoadVector256(ptrB + i + step);
                        var d1 = Avx.Subtract(a1, b1);
                        acc256 = Fma.IsSupported ? Fma.MultiplyAdd(d1, d1, acc256) : Avx.Add(acc256, Avx.Multiply(d1, d1));

                        // Vector256<float> vu = Avx.LoadVector256(ptrA + i);
                        // Vector256<float> vv = Avx.LoadVector256(ptrB + i);
                        // Vector256<float> diff = Avx.Subtract(vu, vv);
                        // Vector256<float> dist = Avx.Multiply(diff, diff);
                        // acc256 = Fma.IsSupported ? Fma.MultiplyAdd(diff, diff, acc256) : Avx.Add(acc256, dist);
                    }
                    for (; i < stop; i += step)
                    {
                        var a0 = Avx.LoadVector256(ptrA + i);
                        var b0 = Avx.LoadVector256(ptrB + i);
                        var d0 = Avx.Subtract(a0, b0);
                        acc256 = Fma.IsSupported ? Fma.MultiplyAdd(d0, d0, acc256) : Avx.Add(acc256, Avx.Multiply(d0, d0));
                    }

                    Vector128<float> lower = acc256.GetLower();
                    Vector128<float> upper = Avx.ExtractVector128(acc256, 1);
                    Vector128<float> sum128 = Sse.Add(lower, upper);
                    sum128 = Sse3.HorizontalAdd(sum128, sum128);
                    sum128 = Sse3.HorizontalAdd(sum128, sum128);
                    float partialSum = sum128.ToScalar();

                    // Vector256<float> temp = Avx.HorizontalAdd(acc256, acc256);
                    // temp = Avx.HorizontalAdd(temp, temp);
                    // float partialSum = temp.GetElement(0) + temp.GetElement(1);

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

    public class EuclideanMetric
    {
        public static float Compute(float[] a, float[] b)
        {
            return (float)Math.Sqrt(SquaredEuclideanMetric.Compute(a, b));
        }
    }
}
