namespace Exports;

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using HNSWIndex;
using HNSWIndex.Metrics;

public static unsafe class HNSWIndexExport
{
    private static string? _lastError;
    private static void SetError(Exception ex) => _lastError = ex.ToString();

    private static nint MakeHandle(HNSWIndex<float[], float> obj) => GCHandle.ToIntPtr(GCHandle.Alloc(obj, GCHandleType.Normal));

    //TODO: Handle multi instance scenario
    private static HNSWParameters<float> _parameters = new();

    private static HNSWIndex<float[], float> Get(nint h)
    {
        if (h == 0) throw new InvalidOperationException("Invalid handle");
        var gch = GCHandle.FromIntPtr(h);
        var index = gch.Target as HNSWIndex<float[], float>;
        if (index is null) throw new InvalidOperationException("Invalid handle");
        return index;
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_get_last_error_utf8", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int GetLastErrorUtf8(byte* buf, int bufLen)
    {
        var s = _lastError ?? string.Empty;
        var need = System.Text.Encoding.UTF8.GetByteCount(s);
        if (bufLen > 0 && buf != null)
        {
            int toWrite = Math.Max(0, bufLen - 1);
            int written = System.Text.Encoding.UTF8.GetBytes(s, new Span<byte>(buf, Math.Min(need, toWrite)));
            buf[written] = 0; // null terminate string
        }
        return need;
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_create", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static nint Create()
    {
        try
        {
            var index = new HNSWIndex<float[], float>(SquaredEuclideanMetric.Compute, _parameters);
            _parameters = new(); // reset parameters for next instance
            return MakeHandle(index);
        }
        catch (Exception ex) { SetError(ex); return 0; }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_free", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static void Free(nint handle)
    {
        if (handle == 0) return;
        try { GCHandle.FromIntPtr(handle).Free(); }
        catch (Exception ex) { SetError(ex); }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_add", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int Add(nint handle, float* vectors, int count, int dim, int* outIds)
    {
        if (handle == 0) return 0;
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
    public static int Remove(nint handle, int* ids, int count)
    {
        if (handle == 0) return 0;
        try
        {
            if (ids == null || count <= 0) return 0;

            var list = new List<int>(count);
            for (int i = 0; i < count; i++) list.Add(ids[i]);

            Get(handle).Remove(list);
            return 0;
        }
        catch (Exception ex) { SetError(ex); return -1; }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_knn_query", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int KnnQuery(nint handle, float* vectors, int count, int dim, int k, int* outIds, float* outDists)
    {
        if (handle == 0) return 0;
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
        if (handle == 0) return 0;
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

    [UnmanagedCallersOnly(EntryPoint = "hnsw_set_collection_size", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SetCollectionSize(int collectionSize)
    {
        try { _parameters.CollectionSize = collectionSize; return 0; }
        catch (Exception ex) { SetError(ex); return -1; }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_set_max_edges", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SetMaxEdges(int maxEdges)
    {
        try { _parameters.MaxEdges = maxEdges; return 0; }
        catch (Exception ex) { SetError(ex); return -1; }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_set_max_candidates", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SetMaxCandidates(int maxCandidates)
    {
        try { _parameters.MaxCandidates = maxCandidates; return 0; }
        catch (Exception ex) { SetError(ex); return -1; }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_set_distribution_rate", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SetDistributionRate(float distRate)
    {
        try { _parameters.DistributionRate = distRate; return 0; }
        catch (Exception ex) { SetError(ex); return -1; }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_set_random_seed", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SetRandomSeed(int seed)
    {
        try { _parameters.RandomSeed = seed; return 0; }
        catch (Exception ex) { SetError(ex); return -1; }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_set_min_nn", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SetMinNN(int minNN)
    {
        try { _parameters.MinNN = minNN; return 0; }
        catch (Exception ex) { SetError(ex); return -1; }
    }
}
