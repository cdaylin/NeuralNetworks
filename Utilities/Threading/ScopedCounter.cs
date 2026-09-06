using Daylin.Utilities.Disposable;

namespace Daylin.Utilities.Threading;

/// <summary>
/// Generalized scoped counter, for detecting reentrancy, counting in-progress operations, tracking resource
/// utilization, etc.
/// </summary>
/// <remarks>
/// This class supports concurrent and nested counting. All operations are thread-safe and non-blocking.
/// </remarks>
public class ScopedCounter
{
    #region Construction

    public ScopedCounter()
    {
    }

    #endregion

    #region Public

    public int Count => count;

    /// <summary>
    /// Increments <see cref="Count"/>.
    /// </summary>
    /// <returns>
    /// Returns an object that will decrement <see cref="Count"/> when disposed.
    /// </returns>
    public IDisposable EnterScope()
    {
        IncrementCount();

        return new ScopedCounterToken(this);
    }

    #endregion

    #region Protected

    protected virtual void OnLockCountIncremented(int lockCount)
    {
    }

    protected virtual void OnLockCountDecremented(int lockCount)
    {
    }

    #endregion

    #region Private

    private volatile int count;

    private void IncrementCount()
    {
        OnLockCountIncremented(Interlocked.Increment(ref count));
    }

    private void DecrementLockCount()
    {
        OnLockCountDecremented(Interlocked.Decrement(ref count));
    }

    private sealed class ScopedCounterToken : DisposableObject
    {
        public ScopedCounterToken(ScopedCounter counter)
        {
            Counter = counter;
        }

        protected override void ReleaseResources()
        {
            Counter.DecrementLockCount();
        }

        private ScopedCounter Counter { get; }
    }

    #endregion
}