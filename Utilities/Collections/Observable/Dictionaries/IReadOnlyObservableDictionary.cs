using System.Collections.Specialized;
using System.ComponentModel;

namespace Daylin.Utilities.Collections.Observable.Dictionaries;

public interface IReadOnlyObservableDictionary<TKey, TValue>
    : IReadOnlyDictionary<TKey, TValue>, INotifyCollectionChanged, INotifyPropertyChanged
{
}