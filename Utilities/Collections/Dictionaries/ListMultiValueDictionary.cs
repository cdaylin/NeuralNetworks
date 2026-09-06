using System.Collections.Immutable;

namespace Daylin.Utilities.Collections.Dictionaries;

public class ListMultiValueDictionary<TKey, TValue>
    : MultiValueDictionary<TKey, TValue, List<TValue>>, IListDictionary<TKey, TValue>
    where TKey : notnull
{
    public ListMultiValueDictionary(
        IEqualityComparer<TKey>? keyComparer = null)
        : base(keyComparer)
    {
    }

    public new IReadOnlyList<TValue> GetValues(TKey key)
    {
        key.ThrowIfNull();

        if (TryGetValuesCollection(key, out List<TValue>? values))
            return values!;

        return ImmutableList<TValue>.Empty;
    }

    public override void AddRange(TKey key, IEnumerable<TValue> values)
    {
        key.ThrowIfNull();
        values.ThrowIfNull();

        if (!values.Any())
            return;

        GetOrCreateValuesCollection(key).AddRange(values);
    }

    protected override List<TValue> CreateValuesCollection()
    {
        return [];
    }
}
