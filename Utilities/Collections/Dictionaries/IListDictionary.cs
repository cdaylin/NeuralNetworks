namespace Daylin.Utilities.Collections.Dictionaries;

public interface IListDictionary<TKey, TValue>
    : IMultiValueDictionary<TKey, TValue>
    where TKey : notnull
{
    new IReadOnlyList<TValue> GetValues(TKey key);
}
