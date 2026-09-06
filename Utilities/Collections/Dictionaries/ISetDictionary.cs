namespace Daylin.Utilities.Collections.Dictionaries;

public interface ISetDictionary<TKey, TValue>
    : IMultiValueDictionary<TKey, TValue>
    where TKey : notnull
{
    new bool Add(TKey key, TValue value);

    new IReadOnlySet<TValue> GetValues(TKey key);
}
