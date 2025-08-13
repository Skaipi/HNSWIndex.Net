namespace HNSWIndex
{
    /// <summary>
    /// Lock immediate neighbors of node in the graph.
    /// Re-entrant per thread: a thread may acquire the same region multiple times.
    /// </summary>
    internal class GraphRegionLocker
    {
        private readonly object bitmapLock = new object();

        // Per-node ownership and reentrancy count.
        private readonly List<int> owner;
        private readonly List<int> count;

        public GraphRegionLocker(int initCapacity)
        {
            owner = new List<int>(initCapacity);
            count = new List<int>(initCapacity);
            if (initCapacity > 0)
            {
                owner.AddRange(new int[initCapacity]);
                count.AddRange(new int[initCapacity]);
            }
        }

        /// <summary>
        /// Extend capacity of the internal arrays. Thread-safe.
        /// </summary>
        public void UpdateCapacity(int newCapacity)
        {
            lock (bitmapLock)
            {
                if (newCapacity <= owner.Count) return;
                int delta = newCapacity - owner.Count;
                owner.AddRange(new int[delta]);
                count.AddRange(new int[delta]);
            }
        }

        public List<IDisposable> FullNodeNeighbourhoodLock(Node node)
        {
            var result = new List<IDisposable>(node.MaxLayer + 1);
            for (int layer = node.MaxLayer; layer >= 0; layer--)
                result.Add(LockNodeNeighbourhood(node, layer));
            return result;
        }

        /// <summary>
        /// Acquire regional lock around node at specific layer.
        /// </summary>
        public IDisposable LockNodeNeighbourhood(Node node, int layer)
        {
            while (true)
            {
                // Get snapshot
                if (!GetNeighbourhoodSnapshot(node, layer, out var s0)) continue;

                int tid = Thread.CurrentThread.ManagedThreadId;

                // Mark neighborhood as busy (or re-enter if already ours).
                lock (bitmapLock)
                {
                    while (!AllFreeLock(s0, tid)) Monitor.Wait(bitmapLock);
                    MarkLock(s0, tid);
                }

                // Validate neighborhood with a second snapshot.
                if (!GetNeighbourhoodSnapshot(node, layer, out var s1))
                {
                    Release(s0); // roll back
                    continue;
                }

                var extras = Except(s1, s0);
                var removed = Except(s0, s1);

                lock (bitmapLock)
                {
                    // Attempt to expand to extras.
                    if (AllFreeLock(extras, tid))
                    {
                        MarkLock(extras, tid);

                        // Optionally shrink immediately to avoid blocking on nodes no longer in s1
                        if (removed.Count != 0) UnmarkLock(removed, tid);

                        // We now hold exactly s1. Wake potential waiters if we shrank.
                        if (removed.Count != 0) Monitor.PulseAll(bitmapLock);

                        return new Releaser(this, s1);
                    }

                    // Cannot expand because some extra id is busy by another thread → roll back s0 and retry.
                    UnmarkLock(s0, tid);
                    Monitor.PulseAll(bitmapLock);
                }
            }
        }

        /// <summary>
        /// Collect ids of all neighbors of a node. Throw of this method is equivalent with returning false.
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
                foreach (var o in outs) result[k++] = o;
                foreach (var i in ins) result[k++] = i;

                ids = result;
                return true;
            }
            catch (Exception ex) when (ex is InvalidOperationException || ex is IndexOutOfRangeException)
            {
                // Enumeration torn by concurrent adjacency edits elsewhere; just retry.
                ids = Array.Empty<int>();
                return false;
            }
        }

        /// <summary>
        /// Returns elements of a that are not in b (no duplicates).
        /// </summary>
        private static List<int> Except(int[] a, int[] b)
        {
            var res = new List<int>(a.Length);
            var setB = new HashSet<int>(b);
            for (int i = 0; i < a.Length; i++) if (!setB.Contains(a[i])) res.Add(a[i]);
            return res;
        }

        /// <summary>
        /// Check if neighborhood is free for the given thread (free or already owned by this thread).
        /// </summary>
        private bool AllFreeLock(IReadOnlyList<int> ids, int tid)
        {
            for (int i = 0; i < ids.Count; i++)
            {
                int id = ids[i];
                int own = owner[id];
                if (own != 0 && own != tid) return false;
            }
            return true;
        }

        /// <summary>
        /// Mark neighborhood as owned by the given thread (re-entrant: increments per-node count if already owned).
        /// bitmapLock must be held by the caller.
        /// </summary>
        private void MarkLock(IReadOnlyList<int> ids, int tid)
        {
            for (int i = 0; i < ids.Count; i++)
            {
                int id = ids[i];
                if (owner[id] == 0)
                {
                    owner[id] = tid;
                    count[id] = 1;
                }
                else
                {
                    // Either re-entrance by same thread, or logic bug.
                    if (owner[id] != tid)
                        throw new InvalidOperationException("MarkLock: ownership conflict detected.");
                    checked { count[id]++; }
                }
            }
        }

        /// <summary>
        /// Unmark neighborhood for the current thread (decrements per-node count; frees on zero).
        /// bitmapLock must be held by the caller.
        /// </summary>
        private void UnmarkLock(IReadOnlyList<int> ids, int tid)
        {
            for (int i = 0; i < ids.Count; i++)
            {
                int id = ids[i];
                if (owner[id] == 0)
                {
                    continue;
                }
                if (owner[id] != tid)
                {
                    // Another thread owns it; this should not happen.
                    throw new InvalidOperationException("UnmarkLock: attempted to release a region owned by a different thread.");
                }

                int c = count[id] - 1;
                if (c <= 0)
                {
                    owner[id] = 0;
                    count[id] = 0;
                }
                else
                {
                    count[id] = c;
                }
            }
        }

        private void Release(int[] ids)
        {
            lock (bitmapLock)
            {
                UnmarkLock(ids, Thread.CurrentThread.ManagedThreadId);
                Monitor.PulseAll(bitmapLock);
            }
        }

        private sealed class Releaser : IDisposable
        {
            private GraphRegionLocker ownerRef;
            private int[] ids;

            public Releaser(GraphRegionLocker owner, int[] ids)
            {
                this.ownerRef = owner;
                this.ids = ids;
            }

            public void Dispose()
            {
                var ownerLocal = Interlocked.Exchange(ref this.ownerRef!, null);
                if (ownerLocal == null) return; // already disposed

                lock (ownerLocal.bitmapLock)
                {
                    ownerLocal.UnmarkLock(ids, Thread.CurrentThread.ManagedThreadId);
                    Monitor.PulseAll(ownerLocal.bitmapLock);
                }

                // GC improvement
                ids = Array.Empty<int>();
            }
        }
    }
}
