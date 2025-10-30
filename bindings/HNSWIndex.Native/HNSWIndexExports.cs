namespace Exports;

using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using HNSWIndex;
using HNSWIndex.Metrics;

public static unsafe class HNSWIndexExport
{
    private static readonly ConcurrentDictionary<nint, HNSWIndex<float[], float>> _handles = new();
    private static long _nextId = 1;

    [ThreadStatic] private static string? _lastError;
    private static void SetError(Exception ex) => _lastError = ex.ToString();

    private static nint AddHandle(HNSWIndex<float[], float> obj)
    {
        var id = (nint)Interlocked.Increment(ref _nextId);
        _handles[id] = obj;
        return id;
    }

    private static HNSWIndex<float[], float> Get(nint h) => _handles.TryGetValue(h, out var obj) ? obj : throw new InvalidOperationException("Invalid handle");

    [UnmanagedCallersOnly(EntryPoint = "hnsw_get_last_error_utf8", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int GetLastErrorUtf8(byte* buf, int bufLen)
    {
        var s = _lastError ?? string.Empty;
        var need = System.Text.Encoding.UTF8.GetByteCount(s);
        if (bufLen > 0 && buf != null)
        {
            var written = System.Text.Encoding.UTF8.GetBytes(s, new Span<byte>(buf, Math.Min(bufLen, need)));
            if (written < bufLen) buf[written] = 0;
        }
        return need;
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_create", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static nint Create()
    {
        try
        {
            var index = new HNSWIndex<float[], float>(SquaredEuclideanMetric.Compute, null);
            return AddHandle(index);
        }
        catch (Exception ex) { SetError(ex); return 0; }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_free", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static void Free(nint handle)
    {
        _handles.TryRemove(handle, out _);
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_add", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int Add(nint handle, float* vectors, int count, int dim, int* outIds)
    {
        try
        {
            if (vectors == null || count <= 0 || dim <= 0) return 0;

            var items = new List<float[]>(count);
            for (int i = 0; i < count; i++)
            {
                float* p = vectors + (i * dim);
                var arr = new float[dim];
                new Span<float>(p, dim).CopyTo(arr.AsSpan());
                items.Add(arr);
            }

            var ids = Get(handle).Add(items);
            int n = Math.Min(count, ids.Length);

            // Write IDs back to caller
            for (int i = 0; i < n; i++) outIds[i] = ids[i];
            return n;
        }
        catch (Exception ex) { SetError(ex); return -1; }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_remove", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static void Remove(nint handle, int* ids, int count)
    {
        try
        {
            if (ids == null || count <= 0) return;

            var list = new List<int>(count);
            for (int i = 0; i < count; i++) list.Add(ids[i]);

            Get(handle).Remove(list); // Uses managed batch Remove
        }
        catch (Exception ex) { SetError(ex); }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_knn_query", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int KnnQuery(nint handle, float* vectors, int count, int dim, int k, int* outIds, float* outDists)
    {
        try
        {
            var queries = new List<float[]>(count);
            for (int i = 0; i < count; i++)
            {
                float* p = vectors + (i * dim);
                var arr = new float[dim];
                new Span<float>(p, dim).CopyTo(arr.AsSpan());
                queries.Add(arr);
            }

            var batchResult = Get(handle).BatchKnnQuery(queries, k);
            for (int i = 0; i < count; i++)
            {
                var res = batchResult[i];
                var n = Math.Min(k, res.Count);
                for (int j = 0; j < n; j++)
                {
                    outIds[i * k + j] = res[j].Id;
                    outDists[i * k + j] = res[j].Distance;
                }
                for (int j = n; j < k; j++) { outIds[i * k + j] = -1; outDists[i * k + j] = float.NaN; }
            }
            return 0;
        }
        catch (Exception ex) { SetError(ex); return -1; }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_range_query", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int RangeQuery(nint handle, float* vectors, int count, int dim, float range, void** outIds, void** outDists, int* counts)
    {
        try
        {
            var queries = new List<float[]>(count);
            for (int i = 0; i < count; i++)
            {
                float* p = vectors + (i * dim);
                var arr = new float[dim];
                new Span<float>(p, dim).CopyTo(arr.AsSpan());
                queries.Add(arr);
            }

            var batchResult = Get(handle).BatchRangeQuery(queries, range);
            for (int i = 0; i < count; i++)
            {
                var res = batchResult[i];
                var n = res.Count;

                int* ids = n > 0 ? (int*)Marshal.AllocHGlobal(sizeof(int) * n) : null;
                float* dists = n > 0 ? (float*)Marshal.AllocHGlobal(sizeof(float) * n) : null;

                for (int j = 0; j < n; j++)
                {
                    ids[j] = res[j].Id;
                    dists[j] = res[j].Distance;
                }
                outIds[i] = ids;
                outDists[i] = dists;
                counts[i] = n;
            }

            return 0;
        }
        catch (Exception ex)
        {
            for (int i = 0; i < count; i++)
            {
                if (outIds != null && outIds[i] != null) { Marshal.FreeHGlobal((nint)outIds[i]); outIds[i] = null; }
                if (outDists != null && outDists[i] != null) { Marshal.FreeHGlobal((nint)outDists[i]); outDists[i] = null; }
                if (counts != null) counts[i] = 0;
            }
            SetError(ex); return -1;
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_free_results", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static unsafe void FreeRangeResults(void** idsArray, void** distsArray, int count)
    {
        if (idsArray == null && distsArray == null) return;

        for (int i = 0; i < count; i++)
        {
            if (idsArray != null && idsArray[i] != null)
            {
                Marshal.FreeHGlobal((nint)idsArray[i]);
                idsArray[i] = null;
            }
            if (distsArray != null && distsArray[i] != null)
            {
                Marshal.FreeHGlobal((nint)distsArray[i]);
                distsArray[i] = null;
            }
        }
    }

    // NOTE: Current protobuf does not support native AoT 
    // [UnmanagedCallersOnly(EntryPoint = "hnsw_serialize", CallConvs = new[] { typeof(CallConvCdecl) })]
    // public static int Serialize(nint handle, byte* pathUtf8, int len)
    // {
    //     try
    //     {
    //         var path = System.Text.Encoding.UTF8.GetString(new ReadOnlySpan<byte>(pathUtf8, len));
    //         Get(handle).Serialize(path);
    //         return 0;
    //     }
    //     catch (Exception ex) { SetError(ex); return -1; }
    // }

    // [UnmanagedCallersOnly(EntryPoint = "hnsw_deserialize", CallConvs = new[] { typeof(CallConvCdecl) })]
    // public static nint Deserialize(byte* pathUtf8, int len)
    // {
    //     try
    //     {
    //         var path = System.Text.Encoding.UTF8.GetString(new ReadOnlySpan<byte>(pathUtf8, len));
    //         var idx = HNSWIndex<float[], float>.Deserialize(SquaredEuclideanMetric.Compute, path);
    //         return AddHandle(idx);
    //     }
    //     catch (Exception ex) { SetError(ex); return -1; }
    // }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_set_collection_size", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SetCollectionSize(nint handle, int collectionSize)
    {
        try { Get(handle).SetCollectionSize(collectionSize); return 0; }
        catch (Exception ex) { SetError(ex); return -1; }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_set_max_edges", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SetMaxEdges(nint handle, int maxEdges)
    {
        try { Get(handle).SetMaxEdges(maxEdges); return 0; }
        catch (Exception ex) { SetError(ex); return -1; }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_set_max_candidates", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SetMaxCandidates(nint handle, int MaxCandidates)
    {
        try { Get(handle).SetMaxCandidates(MaxCandidates); return 0; }
        catch (Exception ex) { SetError(ex); return -1; }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_set_distribution_rate", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SetDistributionRate(nint handle, float distRate)
    {
        try { Get(handle).SetDistributionRate(distRate); return 0; }
        catch (Exception ex) { SetError(ex); return -1; }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_set_random_seed", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SetRandomSeed(nint handle, int seed)
    {
        try { Get(handle).SetRandomSeed(seed); return 0; }
        catch (Exception ex) { SetError(ex); return -1; }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_set_min_nn", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SetMinNN(nint handle, int minNN)
    {
        try { Get(handle).SetMinNN(minNN); return 0; }
        catch (Exception ex) { SetError(ex); return -1; }
    }
}
