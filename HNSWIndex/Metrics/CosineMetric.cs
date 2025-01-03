using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;

namespace HNSWIndex.Metrics
{
    public class CosineMetric
    {
        // NOTE: We assume a and b have the same dimension
        public static unsafe float Compute(float[] a, float[] b)
        {
            int i = 0;
            int length = a.Length;
            float dotScalar = 0f;
            float normAScalar = 0f;
            float normBScalar = 0f;

            fixed (float* pa = a, pb = b)
            {
                float* ptrA = pa;
                float* ptrB = pb;

                if (Avx.IsSupported)
                {
                    Vector256<float> dotAcc = Vector256<float>.Zero;
                    Vector256<float> normAAcc = Vector256<float>.Zero;
                    Vector256<float> normBAcc = Vector256<float>.Zero;

                    int step = Vector256<float>.Count;
                    int stop = length - step + 1;

                    for (; i < stop; i += step)
                    {
                        Vector256<float> va = Avx.LoadVector256(pa + i);
                        Vector256<float> vb = Avx.LoadVector256(pb + i);

                        Vector256<float> prod = Avx.Multiply(va, vb);
                        dotAcc = Avx.Add(dotAcc, prod);

                        Vector256<float> sqA = Avx.Multiply(va, va);
                        normAAcc = Avx.Add(normAAcc, sqA);

                        Vector256<float> sqB = Avx.Multiply(vb, vb);
                        normBAcc = Avx.Add(normBAcc, sqB);
                    }
                    dotScalar = HorizontalSum256(dotAcc);
                    normAScalar = HorizontalSum256(normAAcc);
                    normBScalar = HorizontalSum256(normBAcc);
                }
                else if (Sse.IsSupported)
                {
                    Vector128<float> dotAcc = Vector128<float>.Zero;
                    Vector128<float> normAAcc = Vector128<float>.Zero;
                    Vector128<float> normBAcc = Vector128<float>.Zero;

                    int step = Vector128<float>.Count;
                    int stop = length - step + 1;

                    for (; i < stop; i += step)
                    {
                        Vector128<float> va = Sse.LoadVector128(pa + i);
                        Vector128<float> vb = Sse.LoadVector128(pb + i);

                        Vector128<float> prod = Sse.Multiply(va, vb);
                        dotAcc = Sse.Add(dotAcc, prod);

                        Vector128<float> sqA = Sse.Multiply(va, va);
                        normAAcc = Sse.Add(normAAcc, sqA);

                        Vector128<float> sqB = Sse.Multiply(vb, vb);
                        normBAcc = Sse.Add(normBAcc, sqB);
                    }

                    dotScalar = HorizontalSum128(dotAcc);
                    normAScalar = HorizontalSum128(normAAcc);
                    normBScalar = HorizontalSum128(normBAcc);
                }
                else
                {
                    for (; i < length; i++)
                    {
                        float va = pa[i];
                        float vb = pb[i];
                        dotScalar += va * vb;
                        normAScalar += va * va;
                        normBScalar += vb * vb;
                    }
                }
            }

            float denom = (float)(Math.Sqrt(normAScalar) * Math.Sqrt(normBScalar));
            if (denom < 1e-30f)
                return 1f;
            return 1f - dotScalar / denom;
        }

        // NOTE: We assume a and b have the same dimension and normalized
        // NOTE: Accumulation error for high-dimensional vectors might be signifficant
        public static unsafe float UnitCompute(float[] a, float[] b)
        {
            int i = 0;
            int length = a.Length;
            float dotScalar = 0f;

            fixed (float* pa = a, pb = b)
            {
                if (Avx.IsSupported)
                {
                    Vector256<float> dotAcc = Vector256<float>.Zero;

                    int step = Vector256<float>.Count;
                    int stop = length - step + 1;

                    for (; i < stop; i += step)
                    {
                        Vector256<float> va = Avx.LoadVector256(pa + i);
                        Vector256<float> vb = Avx.LoadVector256(pb + i);
                        Vector256<float> prod = Avx.Multiply(va, vb);
                        dotAcc = Avx.Add(dotAcc, prod);
                    }
                    dotScalar = HorizontalSum256(dotAcc);
                }
                else if (Sse.IsSupported)
                {
                    Vector128<float> dotAcc = Vector128<float>.Zero;

                    int step = Vector128<float>.Count;
                    int stop = length - step + 1;

                    for (; i < stop; i += step)
                    {
                        Vector128<float> va = Sse.LoadVector128(pa + i);
                        Vector128<float> vb = Sse.LoadVector128(pb + i);
                        Vector128<float> prod = Sse.Multiply(va, vb);
                        dotAcc = Sse.Add(dotAcc, prod);
                    }
                    dotScalar = HorizontalSum128(dotAcc);
                }
                for (; i < length; i++)
                {
                    dotScalar += pa[i] * pb[i];
                }
            }
;
            return 1f - dotScalar;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HorizontalSum256(Vector256<float> acc)
        {
            Vector256<float> temp = Avx.HorizontalAdd(acc, acc);
            temp = Avx.HorizontalAdd(temp, temp);
            return temp.GetElement(0) + temp.GetElement(1);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HorizontalSum128(Vector128<float> acc)
        {
            Vector128<float> tmp = Sse.Add(acc, Sse.MoveHighToLow(acc, acc));
            Vector128<float> tmp2 = Sse.Add(tmp, Sse.Shuffle(tmp, tmp, 0x55));
            if (Sse41.IsSupported)
            {
                return Sse41.Extract(tmp2, 0);
            }
            else
            {
                unsafe
                {
                    return Unsafe.ReadUnaligned<float>(&tmp2);
                }
            }
        }
    }
}
