namespace Daylin.Utilities.Threading.Awaiters;

/// <summary>
/// An explicit interface for an awaitable type in the awaitable-awaiter pattern.
/// </summary>
public interface IAwaitable
{
    IAwaiter GetAwaiter();
}
