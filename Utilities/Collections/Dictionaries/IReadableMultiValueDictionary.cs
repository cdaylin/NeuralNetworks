namespace Daylin.Utilities.Collections.Dictionaries;

public interface IReadableMultiValueDictionary<TKey, out TValue>
    where TKey : notnull
{
    IEqualityComparer<TKey> KeyComparer { get; }

    int KeyCount { get; }

    IEnumerable<TKey> Keys { get; }

    int GetValueCount(TKey key);

    IEnumerable<TValue> GetValues(TKey key);

    bool ContainsKey(TKey key);
}
