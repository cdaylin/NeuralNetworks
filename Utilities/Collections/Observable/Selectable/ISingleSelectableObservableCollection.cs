using System.Collections;
using System.Collections.Specialized;
using System.ComponentModel;

namespace Daylin.Utilities.Collections.Observable.Selectable;

public interface ISingleSelectableObservableCollection
    : IList, INotifyCollectionChanged, INotifyPropertyChanged
{
    object? SelectedItem { get; set; }
}

public interface ISingleSelectableReadOnlyObservableCollection<T>
    : ISingleSelectableObservableCollection, IObservableList<T>
    where T : class
{
    new T? SelectedItem { get; set; }

    object? ISingleSelectableObservableCollection.SelectedItem
    {
        get => SelectedItem;
        set => SelectedItem = (T?)value;
    }
}

public interface ISingleSelectableObservableCollection<T>
    : ISingleSelectableReadOnlyObservableCollection<T>, IList<T>
    where T : class
{
}

