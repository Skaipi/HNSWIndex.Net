namespace HNSWIndex
{
    internal class ReallocateEventArgs : EventArgs
    {
        public int NewCapacity { get; }

        public ReallocateEventArgs(int capacity)
        {
            NewCapacity = capacity;
        }
    }
}
