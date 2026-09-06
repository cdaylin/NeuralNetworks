using System.Collections.Specialized;
using System.ComponentModel;

namespace Daylin.Utilities.Collections.Observable;

/// <summary>
/// A list that implements <see cref="INotifyCollectionChanged"/> and <see cref="INotifyPropertyChanged"/>.
/// </summary>
/// <typeparam name="T">
/// Type of items contained in the list.
/// </typeparam>
public interface IObservableList<T>
    : IList<T>, IObservableReadOnlyList<T>
{
}
