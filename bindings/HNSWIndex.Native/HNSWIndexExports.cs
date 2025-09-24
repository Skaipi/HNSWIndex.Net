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
            // TODO: Handle parameters in statefull way.
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
    public static int Add(nint handle, float** vectors, int count, int dim, int* outIds)
    {
        try
        {
            if (vectors == null || count <= 0 || dim <= 0) return 0;

            var items = new List<float[]>(count);
            for (int i = 0; i < count; i++)
            {
                float* p = vectors[i];
                var arr = new float[dim];
                new Span<float>(p, dim).CopyTo(arr);
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
    public static int KnnQuery(nint handle, float* vec, int dim, int k, int* outIds, float* outDists)
    {
        try
        {
            var q = new float[dim];
            new Span<float>(vec, dim).CopyTo(q);
            var res = Get(handle).KnnQuery(q, k);
            var n = Math.Min(k, res.Count);
            for (int i = 0; i < n; i++)
            {
                outIds[i] = res[i].Id;
                outDists[i] = res[i].Distance;
            }
            return n;
        }
        catch (Exception ex) { SetError(ex); return -1; }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_range_query", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int RangeQuery(nint handle, float* vec, int dim, float range, int** outIds, float** outDists)
    {
        try
        {
            var q = new float[dim];
            new Span<float>(vec, dim).CopyTo(q);
            var res = Get(handle).RangeQuery(q, range);
            var n = res.Count;

            int* ids = (int*)Marshal.AllocHGlobal(sizeof(int) * n);
            float* dists = (float*)Marshal.AllocHGlobal(sizeof(float) * n);
            for (int i = 0; i < n; i++)
            {
                ids[i] = res[i].Id;
                dists[i] = res[i].Distance;
            }
            *outIds = ids;
            *outDists = dists;

            return n;
        }
        catch (Exception ex) { SetError(ex); if (outIds != null) *outIds = null; if (outDists != null) *outDists = null; return -1; }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_free_results", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static unsafe void FreeResults(int* ids, float* dists)
    {
        if (ids != null) Marshal.FreeHGlobal((nint)ids);
        if (dists != null) Marshal.FreeHGlobal((nint)dists);
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_serialize", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int Serialize(nint handle, byte* pathUtf8, int len)
    {
        try
        {
            var path = System.Text.Encoding.UTF8.GetString(new ReadOnlySpan<byte>(pathUtf8, len));
            Get(handle).Serialize(path);
            return 0;
        }
        catch (Exception ex) { SetError(ex); return -1; }
    }

    [UnmanagedCallersOnly(EntryPoint = "hnsw_deserialize", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static nint Deserialize(byte* pathUtf8, int len)
    {
        try
        {
            var path = System.Text.Encoding.UTF8.GetString(new ReadOnlySpan<byte>(pathUtf8, len));
            var idx = HNSWIndex<float[], float>.Deserialize(SquaredEuclideanMetric.Compute, path);
            return AddHandle(idx);
        }
        catch (Exception ex) { SetError(ex); return 0; }
    }
}
