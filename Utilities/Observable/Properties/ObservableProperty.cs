using Daylin.Utilities.Events;

using System.ComponentModel;

namespace Daylin.Utilities.Observable.Properties;

/// <summary>
/// Responsible for notifying observers when the value of a property changes.
/// </summary>
/// <remarks>
/// An <see cref="ObservableProperty{T}"/> instance acts as a readable view of a property, providing access to
/// the property value and raising events when the value changes.
/// <para>
/// Two events are raised when the property value changes.  The first event is
/// <see cref="INotifyPropertyChanged.PropertyChanged"/>, with the property name "<see cref="Value"/>".  The second 
/// event is <see cref="IObservableProperty{T}.ValueChanged"/>, which provides access to both the old and new values.
/// </para>
/// </remarks>
/// <typeparam name="T">
/// Type of the property.
/// </typeparam>
public abstract class ObservableProperty<T> : ObservableObject, IObservableProperty<T>
{
    #region Construction

    protected ObservableProperty()
    {
    }

    #endregion

    #region Public

    public event EventHandler<PropertyChangedEventArgs<T>>? ValueChanged
    {
        add => ValueChangedEventManager.Event += value;
        remove => ValueChangedEventManager.Event -= value;
    }

    public T Value
    {
        get
        {
            ThrowExceptionIfDisposed();
            return field;
        }

        protected set
        {
            T oldValue = field;

            if (SetProperty(ref field, value))
                ValueChangedEventManager.RaiseEvent(this, new PropertyChangedEventArgs<T>(oldValue, value));
        }
    } = default!;

    #endregion

    #region Protected

    protected override void ReleaseResources()
    {
        ValueChangedEventManager.Dispose();
        base.ReleaseResources();
    }

    #endregion

    #region Private

    private EventManager<PropertyChangedEventArgs<T>> ValueChangedEventManager { get; } = new();

    #endregion
}
