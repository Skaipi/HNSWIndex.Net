namespace HNSWIndex
{
    internal readonly struct OptionalLock : IDisposable
    {
        private readonly object? _obj;
        private readonly bool _taken;

        public OptionalLock(bool take, object obj)
        {
            if (take)
            {
                Monitor.Enter(obj);
                _taken = true;
                _obj = obj;
            }
            else
            {
                _taken = false;
                _obj = null;
            }
        }

        public void Dispose()
        {
            if (_taken)
                Monitor.Exit(_obj!);
        }
    }
}