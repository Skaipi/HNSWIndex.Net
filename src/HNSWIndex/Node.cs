using System.Runtime.CompilerServices;
using ProtoBuf;

namespace HNSWIndex
{
    [ProtoContract]
    class Node
    {
        [ProtoMember(1)]
        public int Id;

        public object OutEdgesLock { get; } = new();

        public object InEdgesLock { get; } = new();

        [ProtoMember(2)]
        public EdgeList[] OutEdges = Array.Empty<EdgeList>();

        [ProtoMember(3)]
        public EdgeList[] InEdges = Array.Empty<EdgeList>();

        public int MaxLayer => OutEdges.Length - 1;
    }

    [ProtoContract]
    public struct EdgeList
    {
        [ProtoMember(1)]
        public int[] Buffer;
        [ProtoMember(2)]
        public int Count;

        public EdgeList(int capacity)
        {
            Buffer = new int[capacity];
            Count = 0;
        }

        public EdgeList(EdgeList other)
        {
            Buffer = new int[other.Count];
            other.AsSpan().CopyTo(Buffer);
            Count = other.Count;
        }

        public EdgeList(IEnumerable<int> collection)
        {
            Buffer = collection.ToArray();
            Count = Buffer.Length;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<int> AsSpan() => new ReadOnlySpan<int>(Buffer, 0, Count);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<int> AsWritableSpan() => new Span<int>(Buffer, 0, Count);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(int value)
        {
            int next = Count + 1;
            if (Buffer.Length < next)
            {
                Grow(next);
            }

            Buffer[Count] = value;
            Count = next;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Remove(int value)
        {
            var jitBuffer = Buffer;

            for (int i = 0; i < Count; i++)
            {
                if (jitBuffer[i] == value)
                {
                    int last = --Count;
                    if (i != last) jitBuffer[i] = jitBuffer[last];
                    return true;
                }
            }
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void Grow(int needed)
        {
            int current = Buffer?.Length ?? 0;
            int newCapacity = current < 16 ? 16 : current * 2;
            if (newCapacity < needed) newCapacity = needed;

            var newBuf = new int[newCapacity];
            if (Count > 0) Buffer!.AsSpan(0, Count).CopyTo(newBuf);

            Buffer = newBuf;
        }
    }
}