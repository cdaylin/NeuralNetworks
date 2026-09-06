using Daylin.Utilities.Primitives;

using Microsoft.VisualStudio.Threading;

namespace Daylin.Utilities.Threading;

/// <summary>
/// Synchronizes the logic for scheduling an operation to run asynchronously in the thread pool, ensuring that the
/// operation is executed at most one time.
/// </summary>
public class TaskRunner<TResult>
{
    public TaskRunner(Func<Task<TResult>> operation)
    {
        operation.ThrowIfNull();

        Operation = cancellationToken => operation();
    }

    public TaskRunner(Func<CancellationToken, Task<TResult>> operation)
    {
        operation.ThrowIfNull();

        Operation = operation;
    }

    /// <summary>
    /// Schedules the operation to run in the thread pool, if not already scheduled.
    /// </summary>
    /// <remarks>
    /// This method is idempotent and safe for concurrent execution. The operation is scheduled to execute only once
    /// regardless of how many times this method is invoked.
    /// </remarks>
    /// <returns>
    /// Returns a task that represents the scheduled asynchronous operation.
    /// </returns>
    public async Task<TResult> RunAsync()
    {
        await SynchronizationContexts.ThreadPoolContext.ContinueInAsync();

        if (ScheduledTask is not null)
            return await ScheduledTask;

        // Do not pass the cancellation token to the semaphore. Doing so could lead to different results for different
        //  calls to this method, which would violate the guarantee of idempotence.
        using (await Semaphore.EnterAsync())
        {
            ScheduledTask ??= SynchronizationContexts.ThreadPoolContext.EnqueueAsync(
                () => Operation(CancellationTokenSource.Token));
        }

        return await ScheduledTask;
    }

    /// <summary>
    /// Requests cancellation for the operation.
    /// </summary>
    /// <remarks>
    /// If the operation has already completed, this has no effect. If the operation has not yet been scheduled, the
    /// operation will never be scheduled and <see cref="RunAsync"/> will always return a cancelled task. If the
    /// operation has been scheduled and has not yet completed, the cancellation token passed to the operation is
    /// triggered, allowing the operation to cooperatively handle cancellation.
    /// </remarks>
    public void RequestCancellation()
    {
        CancellationTokenSource.Cancel();
    }

    private Func<CancellationToken, Task<TResult>> Operation { get; }

    private CancellationTokenSource CancellationTokenSource { get; } = new CancellationTokenSource();

    private Task<TResult>? ScheduledTask { get; set; }

    private AsyncSemaphore Semaphore { get; } = new AsyncSemaphore(1);
}

/// <summary>
/// Synchronizes the logic for scheduling an operation to run asynchronously in the thread pool, ensuring that the
/// operation is executed at most one time.
/// </summary>
public class TaskRunner : TaskRunner<VoidResult>
{
    public TaskRunner(Func<Task> operation)
        : base(WrapOperation(operation))
    {
        operation.ThrowIfNull();
    }

    public TaskRunner(Func<CancellationToken, Task> operation)
        : base(WrapOperation(operation))
    {
        operation.ThrowIfNull();
    }

    /// <summary>
    /// Schedules the operation to run in the thread pool, if not already scheduled.
    /// </summary>
    /// <remarks>
    /// This method is idempotent and safe for concurrent execution. The operation is scheduled to execute only once
    /// regardless of how many times this method is invoked.
    /// </remarks>
    /// <returns>
    /// Returns a task that tracks the asynchronous execution of the scheduled operation.
    /// </returns>
    public new async Task RunAsync()
    {
        await base.RunAsync();
    }

    private static Func<CancellationToken, Task<VoidResult>> WrapOperation(Func<Task> operation)
    {
        return async _ => { await operation(); return VoidResult.Instance; };
    }

    private static Func<CancellationToken, Task<VoidResult>> WrapOperation(Func<CancellationToken, Task> operation)
    {
        return async cancellationToken => { await operation(cancellationToken); return VoidResult.Instance; };
    }
}