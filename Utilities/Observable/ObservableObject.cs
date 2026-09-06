using Daylin.Utilities.Disposable;

using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace Daylin.Utilities.Observable;

/// <summary>
/// Implements <see cref="INotifyPropertyChanged"/> to provide notifications of changes to public property values.
/// </summary>
public abstract class ObservableObject : DisposableObject, INotifyPropertyChanged
{
    #region Construction

    /// <summary>
    /// Constructs an observable object.
    /// </summary>
    protected ObservableObject()
    {
    }

    #endregion

    #region Public

    /// <summary>
    /// Raised when the value of any public property is changed.
    /// </summary>
    public event PropertyChangedEventHandler? PropertyChanged;

    #endregion

    #region Protected

    protected override void ReleaseResources()
    {
        PropertyChanged = null;

        base.ReleaseResources();
    }

    /// <summary>
    /// Sets the value of a property and raises the <see cref="PropertyChanged"/> event.
    /// </summary>
    /// <remarks>
    /// If the old and new values are equal, then the property is not updated and the <see cref="PropertyChanged"/>
    /// event is not raised.
    /// </remarks>
    /// <returns>
    /// Returns true if and only if the property value was updated.
    /// </returns>
    protected virtual bool SetProperty<T>(
        ref T property,
        T value,
        [CallerMemberName] string propertyName = "")
    {
        if (Equals(property, value))
            return false;

        property = value;

        RaisePropertyChangedEvent(propertyName);

        return true;
    }

    /// <summary>
    /// Raises the <see cref="PropertyChanged"/> event.
    /// </summary>
    /// <remarks>
    /// After all event handlers have completed, <see cref="OnPropertyChangedEventCompleted(string)"/> is called.
    /// </remarks>
    /// <param name="propertyName">
    /// Name of the changed property.
    /// </param>
    protected virtual void RaisePropertyChangedEvent([CallerMemberName] string propertyName = "")
    {
        if (IsDisposed)
            return;

        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));

        OnPropertyChangedEventCompleted(propertyName);
    }

    /// <summary>
    /// Invoked when a property value has been changed, immediately after the <see cref="PropertyChanged"/> event
    /// has been raised and handled.
    /// </summary>
    /// <param name="propertyName">
    /// Name of the changed property.
    /// </param>
    protected virtual void OnPropertyChangedEventCompleted(string propertyName)
    {
    }

    #endregion
}