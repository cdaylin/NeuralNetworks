using Daylin.Utilities.Disposable;

namespace Daylin.Utilities.Events;

/// <summary>
/// A wrapper around an event, allowing an event to be handled as an object.
/// </summary>
/// <remarks>
/// Events in .NET are not objects, so they cannot be passed as arguments or accessed from outside of the
/// declaring type. An <see cref="EventManager{T}"/> object acts as a wrapper around an event, providing a
/// means for encapsulating reusable functionality for events and for providing access to such functionality
/// from outside of the declaring type.
/// <para>
/// This class implements <see cref="IDisposable"/> and calls <see cref="ClearEventHandlers"/> when disposed.
/// Disposing the event wrapper when its declaring object is disposed ensures that registered event handlers will
/// not prevent the event handling objects from being garbage collected. Attempting to add an event handler after
/// disposal will cause an exception to be thrown.
/// </para>
/// </remarks>
/// <typeparam name="T">
/// Event argument type.
/// </typeparam>
public class EventManager<T> : DisposableObject where T : EventArgs
{
    public event EventHandler<T>? Event
    {
        add
        {
            ThrowExceptionIfDisposed();
            PrivateEvent += value;
        }

        remove => PrivateEvent -= value;
    }

    public void RaiseEvent(object? sender, T args)
    {
        ThrowExceptionIfDisposed();

        PrivateEvent?.Invoke(sender, args);
    }

    public void ClearEventHandlers()
    {
        PrivateEvent = null;
    }

    protected override void ReleaseResources()
    {
        ClearEventHandlers();

        base.ReleaseResources();
    }

    private event EventHandler<T>? PrivateEvent;
}
