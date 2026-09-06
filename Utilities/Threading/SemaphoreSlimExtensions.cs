namespace Daylin.Utilities.Threading;

/// <summary>
/// Extension methods for <see cref="SemaphoreSlim"/>.
/// </summary>
public static class SemaphoreSlimExtensions
{
    /// <summary>
    /// Asynchronously waits to enter the semaphore, returning a disposable that releases it.
    /// </summary>
    /// <remarks>
    /// This method enables a clean <c>using</c> pattern for async mutual exclusion:
    /// <code>
    /// private readonly SemaphoreSlim semaphore = new(1, 1);
    ///
    /// public async Task DoWorkAsync()
    /// {
    ///     using (await semaphore.LockAsync())
    ///     {
    ///         await PerformExclusiveOperationAsync();
    ///     }
    /// }
    /// </code>
    /// <para>
    /// <b>Deadlock warning:</b> <see cref="SemaphoreSlim"/> does not support reentrancy. If a thread holding the
    /// lock attempts to acquire it again (directly or through a call chain), a deadlock will occur. Ensure that
    /// code within the lock does not recursively acquire the same semaphore.
    /// </para>
    /// </remarks>
    /// <param name="semaphore">
    /// The semaphore to enter.
    /// </param>
    /// <param name="cancellationToken">
    /// Cancellation token.
    /// </param>
    /// <returns>
    /// A disposable that releases the semaphore when disposed.
    /// </returns>
    public static async Task<SemaphoreLock> LockAsync(
        this SemaphoreSlim semaphore,
        CancellationToken cancellationToken = default)
    {
        semaphore.ThrowIfNull();

        await semaphore.WaitAsync(cancellationToken);

        return new SemaphoreLock(semaphore);
    }
}

/// <summary>
/// A disposable handle that releases a <see cref="SemaphoreSlim"/> when disposed.
/// </summary>
/// <remarks>
/// This struct is returned by <see cref="SemaphoreSlimExtensions.LockAsync"/> to enable the <c>using</c> pattern
/// for async mutual exclusion.
/// <para>
/// <b>Deadlock warning:</b> <see cref="SemaphoreSlim"/> does not support reentrancy. If a thread holding the lock
/// attempts to acquire it again (directly or through a call chain), a deadlock will occur. Ensure that code within
/// the lock does not recursively acquire the same semaphore.
/// </para>
/// </remarks>
public readonly struct SemaphoreLock : IDisposable
{
    private readonly SemaphoreSlim semaphore;

    internal SemaphoreLock(SemaphoreSlim semaphore)
    {
        this.semaphore = semaphore;
    }

    /// <summary>
    /// Releases the semaphore.
    /// </summary>
    public void Dispose()
    {
        semaphore.Release();
    }
}
