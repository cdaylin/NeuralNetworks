using Daylin.Utilities.Threading;

namespace Daylin.Utilities.Extensions;

/// <summary>
/// Extension methods for types <see cref="EventHandler"/> and <see cref="EventHandler{TEventArgs}"/>.
/// </summary>
public static class EventHandlerExtensions
{
    /// <summary>
    /// Calls an event handler asynchronously in a <see cref="ThreadPool"/> thread.
    /// </summary>
    /// <remarks>
    /// This method uses <see cref="SynchronizationContexts.EnqueueAsync(SynchronizationContext, Action)">
    /// SynchronizationContexts.ThreadPoolContext.EnqueueAsync</see> to ensure that the event handler is invoked
    /// asynchronously in the thread pool, even when this method is called from a thread-pool thread.
    /// <para>
    /// Note that when the event handler has multiple targets, the targets are invoked synchronously
    /// (i.e.: sequentially). If one target throws an exception, subsequent targets are not called. To invoke
    /// each target asynchronously and independently of other targets, call <see cref="BroadcastEventAsync"/>.
    /// </para>
    /// </remarks>
    /// <typeparam name="TEventArgs">
    /// Type of event arguments.
    /// </typeparam>
    /// <param name="eventHandler">
    /// Event handler.
    /// </param>
    /// <param name="sender">
    /// Event sender.
    /// </param>
    /// <param name="eventArgs">
    /// Event arguments.
    /// </param>
    /// <returns>
    /// Returns a <see cref="Task"/> representing the asynchronous operation.
    /// </returns>
    /// <seealso cref="BroadcastEventAsync"/>
    public static async Task RaiseEventAsync<TEventArgs>(
        this EventHandler<TEventArgs> eventHandler,
        object? sender,
        TEventArgs eventArgs)
    {
        eventHandler.ThrowIfNull();
        eventArgs.ThrowIfNull();

        await SynchronizationContexts.ThreadPoolContext.EnqueueAsync(
            () => eventHandler.Invoke(sender, eventArgs));
    }

    /// <summary>
    /// Calls an event handler asynchronously in a <see cref="ThreadPool"/> thread.
    /// </summary>
    /// <remarks>
    /// This method uses <see cref="SynchronizationContexts.EnqueueAsync(SynchronizationContext, Action)">
    /// SynchronizationContexts.ThreadPoolContext.EnqueueAsync</see> to ensure that the event handler is invoked
    /// asynchronously in the thread pool, even when this method is called from a thread-pool thread.
    /// <para>
    /// Note that when the event handler has multiple targets, each target is invoked asynchronously, and then all
    /// asynchronous invocations are awaited. Multiple target invocations may execute concurrently. If a target
    /// throws an exception, other targets are still called. If one or more targets throw an exception, this method
    /// throws an <see cref="AggregateException"/> containing those exceptions.
    /// </para>
    /// </remarks>
    /// <typeparam name="TEventArgs">
    /// Type of event arguments.
    /// </typeparam>
    /// <param name="eventHandler">
    /// Event handler.
    /// </param>
    /// <param name="sender">
    /// Event sender.
    /// </param>
    /// <param name="eventArgs">
    /// Event arguments.
    /// </param>
    /// <returns>
    /// Returns a <see cref="Task"/> representing the asynchronous operation.
    /// </returns>
    /// <seealso cref="RaiseEventAsync"/>.
    public static async Task BroadcastEventAsync<TEventArgs>(
        this EventHandler<TEventArgs> eventHandler,
        object? sender,
        TEventArgs eventArgs)
    {
        eventHandler.ThrowIfNull();
        eventArgs.ThrowIfNull();

        List<Task> tasks = [];

        foreach (Delegate handler in eventHandler.GetInvocationList())
            tasks.Add(RaiseEventAsync((EventHandler<TEventArgs>)handler, sender, eventArgs));

        await WhenAllAsync(tasks);
    }

    /// <summary>
    /// Waits for all tasks to complete and throws an <see cref="AggregateException"/> if any of the
    /// tasks faulted.
    /// </summary>
    private static async Task WhenAllAsync(IEnumerable<Task> tasks)
    {
        Task[] taskArray = tasks.ToArray();

        try
        {
            await Task.WhenAll(taskArray);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch
        {
            Exception[] perTaskExceptions = taskArray
                .Where(task => task.IsFaulted && task.Exception is not null)
                .Select(task =>
                {
                    AggregateException aggregateException = task.Exception!;

                    return aggregateException.InnerExceptions.Count == 1
                        ? aggregateException.InnerExceptions[0]
                        : aggregateException;
                })
                .ToArray();

            if (perTaskExceptions.Length == 0)
                throw;

            throw new AggregateException(perTaskExceptions);
        }
    }
}
