using Daylin.Utilities.Threading;

using System.Runtime.CompilerServices;

namespace Daylin.Utilities.Disposable;

/// <summary>
/// Abstract base class providing an implementation of the 'Dispose' pattern.
/// </summary>
/// <seealso cref="AsyncDisposableObject"/>
public abstract class DisposableObject : IDisposable
{
    protected DisposableObject()
    {
    }

    /// <summary>
    /// Determines whether <see cref="Dispose()"/> has been invoked on this object.
    /// </summary>
    public bool IsDisposed => isDisposed.Value;

    private readonly InterlockedBool isDisposed = new();

    public void Dispose()
    {
        if (!isDisposed.TrySetValue(true))
            return;

        ReleaseResources();
    }

    /// <summary>
    /// Derived classes should override this method to release managed resources.
    /// </summary>
    /// <remarks>
    /// <see cref="IsDisposed"/> will be set to <c>true</c> immediately before this method is invoked.
    /// <para>
    /// The recommended pattern for overriding this method is to use a <c>try/finally</c> statement.  Release
    /// managed resources within the <c>try</c> block, and then call the base implementation of this method 
    /// within the <c>finally</c> block.  This pattern allows for base classes to release resources even if a
    /// derived class throws an exception.
    /// </para>
    /// </remarks>
    protected virtual void ReleaseResources()
    {
        // null implementation
    }

    /// <summary>
    /// Throws <see cref="ObjectDisposedException"/> if this object has been disposed.
    /// </summary>
    /// <remarks>
    /// Derived classes may optionally call this method at the beginning of non-private methods to prevent
    /// an object from being used after it has been disposed.
    /// </remarks>
    protected virtual void ThrowExceptionIfDisposed(
        [CallerFilePath] string callerFilePath = "",
        [CallerMemberName] string callerMemberName = "",
        [CallerLineNumber] int callerLineNumber = 0)
    {
        if (IsDisposed)
        {
            throw new ObjectDisposedException(GetType().FullName,
                ObjectDisposedExceptionMessageFormatter.FormatExceptionMessage(
                    GetType(), callerFilePath, callerMemberName, callerLineNumber));
        }
    }
}
