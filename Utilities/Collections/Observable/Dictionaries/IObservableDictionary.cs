namespace Daylin.Utilities.Collections.Observable.Dictionaries;

public interface IObservableDictionary<TKey, TValue>
    : IDictionary<TKey, TValue>, IReadOnlyObservableDictionary<TKey, TValue>
{
}
