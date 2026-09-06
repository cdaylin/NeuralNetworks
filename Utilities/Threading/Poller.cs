using System.Diagnostics;

namespace Daylin.Utilities.Threading;

/// <summary>
/// Provides operations for asynchronously waiting for a condition to be <c>true</c>.
/// </summary>
public class Poller
{
    #region Public Static

    /// <inheritdoc cref="TryWaitForAsync(Func{bool}, TimeSpan, TimeSpan, CancellationToken)"/>
    /// <returns>
    /// Returns a task that represents this asynchronous operation.  Successful completion of the task indicates
    /// that the predicate tested <c>true</c> before the wait time expired.
    /// </returns>
    /// <exception cref="TimeoutException">
    /// Thrown when the wait time expires before the predicate tests <c>true</c>.
    /// </exception>
    /// <exception cref="OperationCanceledException">
    /// Thrown when this operation is cancelled before the predicate tests <c>true</c> or the wait time expires.
    /// </exception>
    public static async Task WaitForAsync(
        Func<bool> predicate,
        TimeSpan pollingInterval,
        TimeSpan waitTime,
        CancellationToken cancellationToken = default)
    {
        await WaitForAsync(
            () => Task.FromResult(predicate()),
            pollingInterval,
            waitTime,
            cancellationToken);
    }

    /// <inheritdoc cref="WaitForAsync(Func{bool}, TimeSpan, TimeSpan, CancellationToken)"/>
    public static async Task WaitForAsync(
        Func<Task<bool>> predicate,
        TimeSpan pollingInterval,
        TimeSpan waitTime,
        CancellationToken cancellationToken = default)
    {
        bool wasSuccessful = await TryWaitForAsync(predicate, pollingInterval, waitTime, cancellationToken);

        if (!wasSuccessful)
        {
            throw new TimeoutException(
                $"The predicate did not test true within the allotted time ({waitTime}).");
        }
    }

    /// <summary>
    /// Asynchronously waits for a condition to be <c>true</c>, periodically testing for the condition.
    /// </summary>
    /// <param name="predicate">
    /// A function that tests for the terminal condition.
    /// </param>
    /// <param name="pollingInterval">
    /// Length of time to wait between consecutive calls to test for the terminal condition.  A negative value
    /// is treated the same as zero.
    /// </param>
    /// <param name="waitTime">
    /// Length of time to continue waiting for the terminal condition.  A negative value is treated the same as
    /// zero.  The predicate will be tested at least once, even if the wait time is non-positive.
    /// </param>
    /// <param name="cancellationToken">
    /// A token for early cancellation of this operation.
    /// </param>
    /// <returns>
    /// Returns <c>true</c> if and only if the predicate tests <c>true</c> before the wait time expires.
    /// </returns>
    /// <exception cref="OperationCanceledException">
    /// Thrown when this operation is cancelled before the predicate tests <c>true</c> or the wait time expires.
    /// </exception>
    public static async Task<bool> TryWaitForAsync(
        Func<bool> predicate,
        TimeSpan pollingInterval,
        TimeSpan waitTime,
        CancellationToken cancellationToken = default)
    {
        return await TryWaitForAsync(
            () => Task.FromResult(predicate()),
            pollingInterval,
            waitTime,
            cancellationToken);
    }

    /// <inheritdoc cref="TryWaitForAsync(Func{bool}, TimeSpan, TimeSpan, CancellationToken)"/>
    public static async Task<bool> TryWaitForAsync(
        Func<Task<bool>> predicate,
        TimeSpan pollingInterval,
        TimeSpan waitTime,
        CancellationToken cancellationToken = default)
    {
        predicate.ThrowIfNull();

        Stopwatch operationTimer = new();
        Stopwatch intervalTimer = new();

        operationTimer.Start();

        while (true)
        {
            intervalTimer.Restart();

            cancellationToken.ThrowIfCancellationRequested();

            if (await predicate())
                return true;

            TimeSpan remainingTimeSpan = waitTime - operationTimer.Elapsed;

            if (remainingTimeSpan <= TimeSpan.Zero)
                return false;

            TimeSpan delayTimeSpan = pollingInterval - intervalTimer.Elapsed;

            if (delayTimeSpan > waitTime)
                delayTimeSpan = waitTime;

            if (delayTimeSpan < TimeSpan.Zero)
                delayTimeSpan = TimeSpan.Zero;

            await Task.Delay(delayTimeSpan, cancellationToken);
        }
    }

    #endregion
}