using System.Diagnostics.CodeAnalysis;

namespace Daylin.Utilities.Threading.Awaiters;

/// <summary>
/// An awaiter that can be used to await a transition to any synchronization context.
/// </summary>
public sealed class SynchronizationContextAwaiter : IAwaiter
{
    public SynchronizationContextAwaiter(SynchronizationContext? synchronizationContext)
    {
        Context = synchronizationContext ?? new();
    }

    public bool IsCompleted => Context.HasThreadAccess();

    [SuppressMessage("Usage", "VSTHRD001:Avoid legacy thread switching APIs", Justification = "Not applicable")]
    public void OnCompleted(Action continuation) => Context.Post(state => continuation(), null);

    public void GetResult() { }

    public SynchronizationContextAwaiter GetAwaiter() => this;

    private SynchronizationContext Context { get; }
}
