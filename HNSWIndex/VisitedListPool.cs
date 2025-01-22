namespace HNSWIndex
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// List of visited nodes. Every element of the list has associated unique version which describe when it was discovered.
    /// </summary>
    internal class VisitedList
    {
        internal short CurrVersion;
        internal short[] Nodes;
        internal int Count => Nodes.Length;

        internal VisitedList(int numElements)
        {
            CurrVersion = 0;
            Nodes = new short[numElements];
        }

        /// <summary>
        /// Marks node as visited in current version of the list.
        /// </summary>
        internal void Add(int nodeId)
        {
            Nodes[nodeId] = CurrVersion;
        }

        /// <summary>
        /// Check if node has been visited.
        /// </summary>
        internal bool Contains(int nodeId)
        {
            return Nodes[nodeId] == CurrVersion;
        }

        /// <summary>
        /// Moves onwership over the list to the new unique version.
        /// </summary>
        internal void Next()
        {
            CurrVersion++;
            // Overflow occurred and version is no longer unique
            if (CurrVersion == 0)
            {
                Array.Clear(Nodes, 0, Nodes.Length);
                CurrVersion++;
            }
        }
    }

    /// <summary>
    /// Pool of VisitedList objects, allowing multiple threads to share them.
    /// Gives exclusive access to a list via GetFreeVisitedList method.
    /// Finally thread should return the list via ReleaseVisitedList.
    /// </summary>
    internal class VisitedListPool
    {
        //TODO: Consider implementing API of using statement
        private readonly Stack<VisitedList> pool;
        private readonly object poolLock = new object();
        private int numElements;
        private int initialSize;

        internal VisitedListPool(int initPoolSize, int elementsAmount)
        {
            numElements = elementsAmount;
            initialSize = initPoolSize;
            pool = new Stack<VisitedList>(initPoolSize);

            for (int i = 0; i < initPoolSize; i++)
            {
                pool.Push(new VisitedList(elementsAmount));
            }
        }


        /// <summary>
        /// Retrieves a VisitedList from the pool if available, otherwise allocates a new one.
        /// </summary>
        internal VisitedList GetFreeVisitedList()
        {
            VisitedList result;
            lock (poolLock)
            {
                if (pool.Count > 0)
                {
                    result = pool.Pop();
                }
                else
                {
                    result = new VisitedList(numElements);
                }
            }

            result.Next();
            return result;
        }

        /// <summary>
        /// Returns a VisitedList to the pool, making it available for reuse.
        /// </summary>
        internal void ReleaseVisitedList(VisitedList visitedList)
        {
            // Do not return list with old capacity
            if (visitedList.Count == numElements)
            {
                lock (poolLock)
                {
                    pool.Push(visitedList);
                }
            }
        }

        /// <summary>
        /// Resize lists in pool to new capacity
        /// </summary>
        internal void Resize(int newCapacity)
        {
            numElements = newCapacity;

            lock (poolLock)
            {
                var currCapacity = Math.Max(pool.Count, initialSize);

                pool.Clear();
                for (int i = 0; i < currCapacity; i++)
                {
                    pool.Push(new VisitedList(newCapacity));
                }
            }
        }
    }
}
