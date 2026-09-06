using System.Collections.Concurrent;

namespace Daylin.Utilities.Collections;

/// <summary>
/// Provides a thread-safe cache of weakly referenced objects created on demand.
/// </summary>
/// <remarks>
/// Cached values are stored using <see cref="WeakReference{T}"/> and are automatically
/// eligible for garbage collection when no strong references remain.
/// </remarks>
/// <typeparam name="TKey">
/// The key type used to identify cached entries. Must be non-nullable.
/// </typeparam>
/// <typeparam name="TValue">
/// The reference type of cached objects stored in the cache.
/// </typeparam>
public sealed class WeakCache<TKey, TValue>
    where TKey : notnull
    where TValue : class
{
    #region Construction

    public WeakCache(Func<TKey, TValue> valueFactory)
    {
        valueFactory.ThrowIfNull();

        ValueFactory = valueFactory;
    }

    #endregion

    #region Public

    public TValue GetOrCreate(TKey key)
    {
        key.ThrowIfNull();

        lock (SyncLock)
        {
            if (Cache.TryGetValue(key, out WeakReference<TValue>? weakReference))
            {
                if (weakReference.TryGetTarget(out TValue? existing))
                    return existing;
            }

            TValue value = ValueFactory(key);

            Cache[key] = new WeakReference<TValue>(value);

            return value;
        }
    }

    public bool TryRemove(TKey key)
    {
        key.ThrowIfNull();

        lock (SyncLock)
            return Cache.TryRemove(key, out _);
    }

    public void Clear()
    {
        lock (SyncLock)
            Cache.Clear();
    }

    #endregion

    #region Private

    private ConcurrentDictionary<TKey, WeakReference<TValue>> Cache { get; }
        = new ConcurrentDictionary<TKey, WeakReference<TValue>>();

    private object SyncLock { get; } = new();

    private Func<TKey, TValue> ValueFactory { get; }

    #endregion
}
