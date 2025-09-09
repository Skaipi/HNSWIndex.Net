
using System.Runtime.CompilerServices;

namespace HNSWIndex
{
    internal struct BinaryHeap<T>
    {
        internal IComparer<T> Comparer;
        internal List<T> Buffer;
        internal bool Any => Buffer.Count > 0;
        internal int Count => Buffer.Count;
        internal BinaryHeap(List<T> buffer) : this(buffer, Comparer<T>.Default) { }
        internal BinaryHeap(List<T> buffer, IComparer<T> comparer)
        {
            Buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
            Comparer = comparer;
            for (int i = 1; i < Buffer.Count; ++i) { SiftUp(i); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void Push(T item)
        {
            Buffer.Add(item);
            SiftUp(Buffer.Count - 1);
        }

        internal T Top()
        {
            return Buffer[0];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal T Pop()
        {
            if (Buffer.Count > 0)
            {
                var result = Buffer[0];

                Buffer[0] = Buffer[Buffer.Count - 1];
                Buffer.RemoveAt(Buffer.Count - 1);
                SiftDown(0);

                return result;
            }

            throw new InvalidOperationException("Heap is empty");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void SiftDown(int i)
        {
            if (Buffer.Count == 0) return;

            var item = Buffer[i];
            var half = Buffer.Count >> 1; // Only need to check until the first non-leaf node

            while (i < half)
            {
                int left = (i << 1) + 1;
                int right = left + 1;
                int maxChild = (right < Buffer.Count && Comparer.Compare(Buffer[left], Buffer[right]) < 0) ? right : left;

                if (Comparer.Compare(Buffer[maxChild], item) <= 0)
                    break;

                Buffer[i] = Buffer[maxChild];
                i = maxChild;
            }

            Buffer[i] = item;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void SiftUp(int i)
        {
            T item = Buffer[i];
            while (i > 0)
            {
                int p = (i - 1) >> 1;
                T parent = Buffer[p];
                if (Comparer.Compare(item, parent) <= 0)
                {
                    break;
                }

                // Move parent down
                Buffer[i] = parent;
                i = p;
            }

            // Place the original item at its correct position
            Buffer[i] = item;
        }
    }
}