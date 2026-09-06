using System.Collections.Immutable;

namespace Daylin.Utilities.Collections.Dictionaries;

public class HashSetMultiValueDictionary<TKey, TValue>
    : MultiValueDictionary<TKey, TValue, HashSet<TValue>>, ISetDictionary<TKey, TValue>
    where TKey : notnull
{
    public HashSetMultiValueDictionary(
        IEqualityComparer<TKey>? keyComparer = null)
        : base(keyComparer)
    {
    }

    public new IReadOnlySet<TValue> GetValues(TKey key)
    {
        key.ThrowIfNull();

        if (TryGetValuesCollection(key, out HashSet<TValue>? values))
            return values!;

        return ImmutableHashSet<TValue>.Empty;
    }

    public new bool Add(TKey key, TValue value)
    {
        key.ThrowIfNull();

        return GetOrCreateValuesCollection(key).Add(value);
    }

    protected override HashSet<TValue> CreateValuesCollection()
    {
        return [];
    }
}
