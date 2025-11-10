
using System.Runtime.CompilerServices;

namespace HNSWIndex
{
    internal struct BinaryHeap<T, TComparer> where TComparer : struct, IComparer<T>
    {
        private TComparer comparer;
        private T[] buffer;
        private int _count;
        internal bool Any => _count > 0;
        internal int Count => _count;

        internal BinaryHeap(int capacity = 0, TComparer cmp = default)
        {
            buffer = capacity > 0 ? new T[capacity] : Array.Empty<T>();
            comparer = cmp;
        }

        internal BinaryHeap(ReadOnlySpan<T> items, TComparer cmp = default)
        {
            _count = items.Length;
            buffer = _count == 0 ? Array.Empty<T>() : items.ToArray();
            comparer = cmp;
            // heapify
            for (int i = (buffer.Length >> 1) - 1; i >= 0; --i) SiftDown(i, buffer[i], _count);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void Push(T item)
        {
            if (_count == buffer.Length) ExtendBuffer();
            SiftUp(_count++, item);
        }

        internal T Peek()
        {
            return buffer[0];
        }

        internal T[] ToArray()
        {
            return buffer[0.._count];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void Clear()
        {
            _count = 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal T Pop()
        {
            int jitCount = _count;
            if (jitCount == 0) throw new InvalidOperationException("Heap is empty");

            var result = buffer[0];
            var lastLeaf = buffer[--jitCount];
            _count = jitCount;

            if (jitCount != 0) SiftDown(0, lastLeaf, jitCount);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void SiftDown(int i, T item, int count)
        {
            var jitBuffer = buffer;
            var jitComparer = comparer;
            var half = count >> 1;

            while (i < half)
            {
                int left = (i << 1) + 1;
                int right = left + 1;
                int maxChild = (right < count && jitComparer.Compare(jitBuffer[left], jitBuffer[right]) < 0) ? right : left;

                if (jitComparer.Compare(jitBuffer[maxChild], item) <= 0) break;

                jitBuffer[i] = jitBuffer[maxChild];
                i = maxChild;
            }

            jitBuffer[i] = item;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void SiftUp(int i, T item)
        {
            var jitBuffer = buffer;
            var jitComparer = comparer;

            while (i > 0)
            {
                int p = (i - 1) >> 1;
                T parent = jitBuffer[p];
                if (jitComparer.Compare(item, parent) <= 0) break;

                // Move parent down and keep current item "hanging"
                jitBuffer[i] = parent;
                i = p;
            }

            // Place the original item at its correct position
            jitBuffer[i] = item;
        }

        private void ExtendBuffer()
        {
            int newSize = buffer.Length == 0 ? 16 : buffer.Length * 2;
            Array.Resize(ref buffer, newSize);
        }
    }
}