using ProtoBuf;

namespace HNSWIndex;

/// <summary>
/// Class implementing an active set of node ids. 
/// The active set is used to keep track of the nodes that are currently in the graph.
/// </summary>
[ProtoContract]
internal class ActiveSet
{
    [ProtoMember(1)]
    private int[] dense;
    [ProtoMember(2)]
    private int[] sparse;
    [ProtoMember(3)]
    private int count;
    private readonly object setLock = new object();

    /// <summary>
    /// Default constructor for the active set. Initializes with zero capacity.
    /// </summary>
    public ActiveSet() : this(0) { }

    /// <summary>
    /// Active set implementation using a dense-sparse approach.
    /// The dense array stores the active ids, while the sparse
    /// array maps ids to their index in the dense array.
    /// </summary>
    internal ActiveSet(int capacity)
    {
        dense = new int[capacity];
        sparse = new int[capacity];
        count = 0;
    }

    /// <summary>
    /// Active set constructor from a list of active ids.
    /// </summary>
    internal ActiveSet(int[] activeIds)
    {
        dense = activeIds;
        count = activeIds.Length;

        sparse = new int[activeIds.Length];
        for (int i = 0; i < count; i++)
        {
            sparse[dense[i]] = i;
        }
    }

    internal int Count => Volatile.Read(ref count);

    /// <summary>
    /// Resize the active set to accommodate at least the given capacity.
    /// </summary>
    internal void EnsureCapacity(int capacity)
    {
        lock (setLock)
        {
            if (dense.Length < capacity)
            {
                Array.Resize(ref dense, capacity);
                Array.Resize(ref sparse, capacity);
            }
        }
    }

    /// <summary>
    /// Add an id to the active set. The id must not be present in the set before.
    /// </summary>
    internal void Add(int id)
    {
        lock (setLock)
        {
            int idx = count++;
            dense[idx] = id;
            sparse[id] = idx;
        }
    }

    /// <summary>
    /// Remove an id from the active set. The id must be present in the set before.
    /// </summary>
    internal void Remove(int id)
    {
        lock (setLock)
        {
            int idx = sparse[id];
            int lastIdx = --count;
            int lastId = dense[lastIdx];

            // move last into removed slot
            dense[idx] = lastId;
            sparse[lastId] = idx;
        }
    }

    internal int[] ActiveIds => dense[..count];

    /// <summary>
    /// Get a snapshot of active ids. The snapshot is a copy of the current
    /// active set and will not reflect future changes.
    /// </summary>
    internal int[] Snapshot()
    {
        lock (setLock)
        {
            var snap = new int[count];
            Array.Copy(dense, 0, snap, 0, count);
            return snap;
        }
    }

    /// <summary>
    /// Check if an id is in the active set under O(1).
    /// </summary>
    internal bool Contains(int id)
    {
        lock (setLock)
        {
            int idx = sparse[id];
            return idx >= 0 && idx < count && dense[idx] == id;
        }
    }
}