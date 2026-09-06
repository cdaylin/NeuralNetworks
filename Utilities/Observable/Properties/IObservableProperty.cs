using System.ComponentModel;

namespace Daylin.Utilities.Observable.Properties;

/// <summary>
/// Responsible for notifying observers when the value of a property changes.
/// </summary>
/// <remarks>
/// Two events are raised when the property value changes.  The first event is
/// <see cref="INotifyPropertyChanged.PropertyChanged"/>, with the property name "<see cref="Value"/>".  The second 
/// event is <see cref="ValueChanged"/>, which provides access to both the old and new values.
/// </remarks>
/// <typeparam name="T">
/// Type of the property.
/// </typeparam>
public interface IObservableProperty<T> : INotifyPropertyChanged, IDisposable
{
    /// <summary>
    /// The property value.
    /// </summary>
    T Value { get; }

    /// <summary>
    /// Raised when <see cref="Value"/> has changed.
    /// </summary>
    /// <remarks>
    /// Unlike <see cref="INotifyPropertyChanged.PropertyChanged"/>, this event provides access to both the old and
    /// new values.
    /// <para>
    /// This event is always raised immediately after <see cref="INotifyPropertyChanged.PropertyChanged"/>.
    /// </para>
    /// </remarks>
    event EventHandler<PropertyChangedEventArgs<T>>? ValueChanged;
}
