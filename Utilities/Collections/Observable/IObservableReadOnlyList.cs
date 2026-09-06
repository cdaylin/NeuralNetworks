using System.Collections.Specialized;
using System.ComponentModel;

namespace Daylin.Utilities.Collections.Observable;

/// <summary>
/// A read-only view of a list that implements <see cref="INotifyCollectionChanged"/> and
/// <see cref="INotifyPropertyChanged"/>.
/// </summary>
/// <typeparam name="T">
/// Type of items contained in the list.
/// </typeparam>
public interface IObservableReadOnlyList<out T>
    : IReadOnlyList<T>, INotifyCollectionChanged, INotifyPropertyChanged
{
}