
namespace HNSWIndex
{
    /// <summary>
    /// Lock immediate neighbors of node in the graph.
    /// </summary>
    internal class GraphRegionLocker
    {
        private readonly object bitmapLock = new object();
        private readonly List<bool> busy;

        public GraphRegionLocker(int initCapacity)
        {
            busy = new List<bool>(initCapacity);
            busy.AddRange(new bool[initCapacity]);
        }

        /// <summary>
        /// Extend capacity of busy flags
        /// </summary>
        public void UpdateCapacity(int newCapacity)
        {
            busy.AddRange(new bool[newCapacity - busy.Count]);
        }

        /// <summary>
        /// Acquire regional lock around node at specific layer
        /// </summary>
        public IDisposable LockNodeNeighbourhood(Node node, int layer)
        {
            while (true)
            {
                // Get snapshot
                if (!GetNeighbourhoodSnapshot(node, layer, out var s0)) continue;

                // Mark neighborhood as busy
                // Adjacency of s0 cannot be modified after this
                lock (bitmapLock)
                {
                    while (!AllFreeLock(s0)) Monitor.Wait(bitmapLock);
                    MarkLock(s0);
                }

                // Take second version to validate neighborhood.
                if (!GetNeighbourhoodSnapshot(node, layer, out var s1))
                {
                    Release(s0);
                    continue;
                }

                var extras = Except(s1, s0);
                var removed = Except(s0, s1);
                lock (bitmapLock)
                {
                    // Attempt to expand to extras
                    if (AllFreeLock(extras))
                    {
                        MarkLock(extras);
                        // Optionally shrink immediately to avoid blocking on nodes no longer in s1
                        if (removed.Count != 0) UnmarkLock(removed);

                        // We now hold exactly s1. Wake potential waiters if we shrank.
                        if (removed.Count != 0) Monitor.PulseAll(bitmapLock);

                        return new Releaser(this, s1);
                    }

                    // Cannot expand because some extra id is busy → roll back s0 and retry.
                    UnmarkLock(s0);
                    Monitor.PulseAll(bitmapLock);
                }
            }
        }

        /// <summary>
        /// Collect ids of all neighbors of a node. Throw of this method is equivalent with returnning false.
        /// </summary>
        private bool GetNeighbourhoodSnapshot(Node node, int layer, out int[] ids)
        {
            try
            {
                var outs = node.OutEdges[layer];
                var ins = node.InEdges[layer];

                // Build [node] + outs + ins
                var result = new int[1 + outs.Count + ins.Count];
                int k = 0;
                result[k++] = node.Id;
                for (int i = 0; i < outs.Count; i++) result[k++] = outs[i];
                for (int i = 0; i < ins.Count; i++) result[k++] = ins[i];

                ids = result;
                return true;
            }
            catch (InvalidOperationException)
            {
                // Enumeration torn by concurrent adjacency edits elsewhere; just retry.
                ids = Array.Empty<int>();
                return false;
            }
        }

        /// <summary>
        /// Returns elements that are not in b (no duplicates).
        /// </summary>
        private static List<int> Except(int[] a, int[] b)
        {
            var res = new List<int>(a.Length);
            var setB = new HashSet<int>(b);
            for (int i = 0; i < a.Length; i++) if (!setB.Contains(a[i])) res.Add(a[i]);
            return res;
        }

        /// <summary>
        /// Check if neighborhood is free.
        /// </summary>
        private bool AllFreeLock(IReadOnlyList<int> ids)
        {
            for (int i = 0; i < ids.Count; i++)
            {
                int id = ids[i];
                if (busy[id]) return false;
            }
            return true;
        }

        /// <summary>
        /// Mark neighborhood as busy
        /// </summary>
        private void MarkLock(IReadOnlyList<int> ids)
        {
            for (int i = 0; i < ids.Count; i++) busy[ids[i]] = true;
        }

        /// <summary>
        /// Mark neighborhood as free
        /// </summary>
        private void UnmarkLock(IReadOnlyList<int> ids)
        {
            for (int i = 0; i < ids.Count; i++) busy[ids[i]] = false;
        }

        private void Release(int[] ids)
        {
            lock (bitmapLock)
            {
                UnmarkLock(ids);
                Monitor.PulseAll(bitmapLock);
            }
        }

        private sealed class Releaser : IDisposable
        {
            private GraphRegionLocker owner;
            private int[] ids;

            public Releaser(GraphRegionLocker owner, int[] ids)
            {
                this.owner = owner;
                this.ids = ids;
            }

            public void Dispose()
            {
                var owner = Interlocked.Exchange(ref this.owner!, null);
                if (owner == null) return; // already disposed

                lock (owner.bitmapLock)
                {
                    // Clear bits and wake waiters
                    owner.UnmarkLock(ids);
                    Monitor.PulseAll(owner.bitmapLock);
                }

                // GC improvement (hopefully)
                ids = Array.Empty<int>();
            }
        }
    }
}