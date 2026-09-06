using System.Runtime.CompilerServices;

namespace Daylin.Utilities.Threading.Awaiters;

/// <summary>
/// An explicit interface for an awaiter in the awaitable-awaiter pattern.
/// </summary>
public interface IAwaiter : INotifyCompletion
{
    bool IsCompleted { get; }

    void GetResult();

    IAwaiter GetAwaiter() => this;
}
