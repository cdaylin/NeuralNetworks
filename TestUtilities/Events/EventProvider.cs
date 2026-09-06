namespace Daylin.TestUtilities.Events;

/// <summary>
/// A class that exposes events and methods for raising those events.
/// </summary>
/// <typeparam name="TEventArgs">
/// Type of argument used by the events raised by this class.
/// </typeparam>
public class EventProvider<TEventArgs> where TEventArgs : EventArgs, new()
{
    #region Public Static

    public static event EventHandler<TEventArgs>? StaticEvent;

    public static void RaiseStaticEvent(TEventArgs? args)
    {
        StaticEvent?.Invoke(null, args ?? new());
    }

    public static int StaticEventHandlerCount => GetStaticEventHandlers().Count();

    public static void ResetStaticEvent()
    {
        foreach (EventHandler<TEventArgs> handler in GetStaticEventHandlers())
            StaticEvent -= handler;
    }

    #endregion

    #region Public

    public event EventHandler<TEventArgs>? InstanceEvent;

    public void RaiseInstanceEvent(TEventArgs? args)
    {
        InstanceEvent?.Invoke(null, args ?? new TEventArgs());
    }

    #endregion

    #region Private Static

    private static IEnumerable<EventHandler<TEventArgs>> GetStaticEventHandlers()
    {
        return StaticEvent?.GetInvocationList().OfType<EventHandler<TEventArgs>>()
            ?? Enumerable.Empty<EventHandler<TEventArgs>>();
    }

    #endregion
}
