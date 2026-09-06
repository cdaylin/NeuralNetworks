using System.Diagnostics.CodeAnalysis;

using Daylin.Utilities.Threading.Awaiters;

namespace Daylin.Utilities.Threading;

/// <summary>
/// Extension and helper methods for working with <see cref="SynchronizationContext"/> objects.
/// </summary>
/// <remarks>
/// <para>
/// <strong>Usage Guidelines</strong>
/// </para>
/// <para>
/// These methods provide a consistent approach for marshalling operations between synchronization contexts.
/// Choose a method based on three decisions:
/// </para>
/// <list type="number">
/// <item>
/// <strong>Which context?</strong> Use <see cref="ThreadPoolContext"/> for background work, or a captured UI
/// context for UI operations.
/// </item>
/// <item>
/// <strong>Always enqueue, or invoke directly if already on context?</strong> Use
/// <see cref="EnqueueAsync(SynchronizationContext, Action)">EnqueueAsync</see> to always post to the queue.
/// Use <see cref="InvokeAsync(SynchronizationContext, Action)">InvokeAsync</see> to execute synchronously when
/// already on the target context (avoiding unnecessary queue overhead).
/// </item>
/// <item>
/// <strong>Await the result, or observe without awaiting?</strong> Await the returned task when you need the
/// result or must wait for completion. Chain <c>.Observe()</c> when the result is not needed but exceptions
/// should still be routed to <see cref="ObservedTaskFaulted"/>.
/// </item>
/// </list>
/// <para>
/// <strong>Method Selection Guide</strong>
/// </para>
/// <list type="table">
/// <listheader>
/// <term>Scenario</term>
/// <description>Pattern</description>
/// </listheader>
/// <item>
/// <term>Post and await result</term>
/// <description><c>await context.EnqueueAsync(() => ...)</c></description>
/// </item>
/// <item>
/// <term>Invoke (sync if on context) and await</term>
/// <description><c>await context.InvokeAsync(() => ...)</c></description>
/// </item>
/// <item>
/// <term>Post without awaiting</term>
/// <description><c>context.EnqueueAsync(() => ...).Observe()</c></description>
/// </item>
/// <item>
/// <term>Invoke without awaiting</term>
/// <description><c>context.InvokeAsync(() => ...).Observe()</c></description>
/// </item>
/// <item>
/// <term>Switch context for remainder of method</term>
/// <description><c>await context.ContinueInAsync()</c></description>
/// </item>
/// </list>
/// <para>
/// <strong>Prefer ContinueInAsync over ConfigureAwait(false)</strong>
/// </para>
/// <para>
/// Use <c>await context.ContinueInAsync()</c> instead of <c>await task.ConfigureAwait(false)</c>. Both prevent
/// capturing the current context, but <see cref="ContinueInAsync"/> is explicit about the target context and
/// continues synchronously if already on that context. <c>ConfigureAwait(false)</c> continues on an arbitrary
/// thread pool thread.
/// </para>
/// <para>
/// <strong>Relationship to Task.Run</strong>
/// </para>
/// <para>
/// <c>Task.Run(() => ...)</c> is equivalent to <c>ThreadPoolContext.EnqueueAsync(() => ...)</c>. Prefer
/// <see cref="EnqueueAsync(SynchronizationContext, Action)">EnqueueAsync</see> for consistency with other
/// context-switching code in this codebase.
/// </para>
/// <para>
/// <strong>Exception Handling</strong>
/// </para>
/// <para>
/// When chaining <c>.Observe()</c>, exceptions (other than <see cref="OperationCanceledException"/>) are routed
/// to the <see cref="ObservedTaskFaulted"/> event. Register a handler for this event to log or display errors
/// from observed tasks. If no handler is registered, exceptions become unobserved task exceptions.
/// </para>
/// </remarks>
public static class SynchronizationContexts
{
    #region Public Static

    /// <summary>
    /// A <see cref="SynchronizationContext"/> that executes operations in the <see cref="ThreadPool"/>.
    /// </summary>
    /// <remarks>
    /// This context can be used with <see cref="ContinueInAsync"/> to switch execution to a thread pool thread, or
    /// with <see cref="EnqueueAsync(SynchronizationContext, Action)"/> to post operations to the thread pool.
    /// </remarks>
    public static SynchronizationContext ThreadPoolContext { get; } = new();

    /// <summary>
    /// Raised when a faulted task is observed via <see cref="Extensions.TaskExtensions.Observe(Task, string, string, int)"/>.
    /// </summary>
    /// <remarks>
    /// This event is raised for exceptions of any type other than <see cref="OperationCanceledException"/>.
    /// </remarks>
    public static event EventHandler<ObservedTaskExceptionEventArgs>? ObservedTaskFaulted;

    /// <summary>
    /// Enqueues an operation to run asynchronously on the specified synchronization context.
    /// </summary>
    /// <remarks>
    /// The operation is always posted to the context's queue, even if already executing on that context.
    /// </remarks>
    /// <param name="context">
    /// The synchronization context on which to run the operation.
    /// </param>
    /// <param name="operation">
    /// The operation to run.
    /// </param>
    /// <returns>
    /// Returns a task that completes when the operation has finished executing.
    /// </returns>
    [SuppressMessage("Usage", "VSTHRD001:Avoid legacy thread switching APIs")]
    public static Task EnqueueAsync(this SynchronizationContext context, Action operation)
    {
        context.ThrowIfNull();
        operation.ThrowIfNull();

        TaskCompletionSource taskCompletionSource = new();

        context.Post(_ =>
        {
            try
            {
                operation();
                taskCompletionSource.SetResult();
            }
            catch (Exception exception)
            {
                taskCompletionSource.SetException(exception);
            }
        }, state: null);

        return taskCompletionSource.Task;
    }

    /// <summary>
    /// Enqueues an asynchronous operation to run on the specified synchronization context.
    /// </summary>
    /// <remarks>
    /// The operation is always posted to the context's queue, even if already executing on that context.
    /// </remarks>
    /// <param name="context">
    /// The synchronization context on which to run the operation.
    /// </param>
    /// <param name="operation">
    /// The asynchronous operation to run.
    /// </param>
    /// <returns>
    /// Returns a task that completes when the operation has finished executing.
    /// </returns>
    [SuppressMessage("Usage", "VSTHRD001:Avoid legacy thread switching APIs")]
    [SuppressMessage("Usage", "VSTHRD101:Avoid unsupported async delegates")]
    public static Task EnqueueAsync(this SynchronizationContext context, Func<Task> operation)
    {
        context.ThrowIfNull();
        operation.ThrowIfNull();

        TaskCompletionSource taskCompletionSource = new();

        context.Post(async _ =>
        {
            try
            {
                await operation();
                taskCompletionSource.SetResult();
            }
            catch (Exception exception)
            {
                taskCompletionSource.SetException(exception);
            }
        }, state: null);

        return taskCompletionSource.Task;
    }

    /// <summary>
    /// Enqueues an operation to run asynchronously on the specified synchronization context.
    /// </summary>
    /// <remarks>
    /// The operation is always posted to the context's queue, even if already executing on that context.
    /// </remarks>
    /// <typeparam name="TResult">
    /// The type of result returned by the operation.
    /// </typeparam>
    /// <param name="context">
    /// The synchronization context on which to run the operation.
    /// </param>
    /// <param name="operation">
    /// The operation to run.
    /// </param>
    /// <returns>
    /// Returns a task that completes with the operation's result when the operation has finished executing.
    /// </returns>
    [SuppressMessage("Usage", "VSTHRD001:Avoid legacy thread switching APIs")]
    public static Task<TResult> EnqueueAsync<TResult>(this SynchronizationContext context, Func<TResult> operation)
    {
        context.ThrowIfNull();
        operation.ThrowIfNull();

        TaskCompletionSource<TResult> taskCompletionSource = new();

        context.Post(_ =>
        {
            try
            {
                TResult result = operation();
                taskCompletionSource.SetResult(result);
            }
            catch (Exception exception)
            {
                taskCompletionSource.SetException(exception);
            }
        }, state: null);

        return taskCompletionSource.Task;
    }

    /// <summary>
    /// Enqueues an asynchronous operation to run on the specified synchronization context.
    /// </summary>
    /// <remarks>
    /// The operation is always posted to the context's queue, even if already executing on that context.
    /// </remarks>
    /// <typeparam name="TResult">
    /// The type of result returned by the operation.
    /// </typeparam>
    /// <param name="context">
    /// The synchronization context on which to run the operation.
    /// </param>
    /// <param name="operation">
    /// The asynchronous operation to run.
    /// </param>
    /// <returns>
    /// Returns a task that completes with the operation's result when the operation has finished executing.
    /// </returns>
    [SuppressMessage("Usage", "VSTHRD001:Avoid legacy thread switching APIs")]
    [SuppressMessage("Usage", "VSTHRD101:Avoid unsupported async delegates")]
    public static Task<TResult> EnqueueAsync<TResult>(this SynchronizationContext context, Func<Task<TResult>> operation)
    {
        context.ThrowIfNull();
        operation.ThrowIfNull();

        TaskCompletionSource<TResult> taskCompletionSource = new();

        context.Post(async _ =>
        {
            try
            {
                TResult result = await operation();
                taskCompletionSource.SetResult(result);
            }
            catch (Exception exception)
            {
                taskCompletionSource.SetException(exception);
            }
        }, state: null);

        return taskCompletionSource.Task;
    }

    /// <summary>
    /// Invokes an operation on the specified synchronization context, executing synchronously if already on that
    /// context.
    /// </summary>
    /// <remarks>
    /// If the current thread is already executing in the target context, the operation is invoked synchronously.
    /// Otherwise, the operation is enqueued via <see cref="EnqueueAsync(SynchronizationContext, Action)"/>.
    /// </remarks>
    /// <param name="context">
    /// The synchronization context on which to run the operation.
    /// </param>
    /// <param name="operation">
    /// The operation to run.
    /// </param>
    /// <returns>
    /// Returns a task that completes when the operation has finished executing.
    /// </returns>
    public static Task InvokeAsync(this SynchronizationContext context, Action operation)
    {
        context.ThrowIfNull();
        operation.ThrowIfNull();

        if (context.HasThreadAccess())
        {
            try
            {
                operation();
                return Task.CompletedTask;
            }
            catch (Exception exception)
            {
                return Task.FromException(exception);
            }
        }

        return context.EnqueueAsync(operation);
    }

    /// <summary>
    /// Invokes an asynchronous operation on the specified synchronization context, executing synchronously if
    /// already on that context.
    /// </summary>
    /// <remarks>
    /// If the current thread is already executing in the target context, the operation is invoked synchronously.
    /// Otherwise, the operation is enqueued via <see cref="EnqueueAsync(SynchronizationContext, Func{Task})"/>.
    /// </remarks>
    /// <param name="context">
    /// The synchronization context on which to run the operation.
    /// </param>
    /// <param name="operation">
    /// The asynchronous operation to run.
    /// </param>
    /// <returns>
    /// Returns a task that completes when the operation has finished executing.
    /// </returns>
    public static Task InvokeAsync(this SynchronizationContext context, Func<Task> operation)
    {
        context.ThrowIfNull();
        operation.ThrowIfNull();

        if (context.HasThreadAccess())
            return operation();

        return context.EnqueueAsync(operation);
    }

    /// <summary>
    /// Invokes an operation on the specified synchronization context, executing synchronously if already on that
    /// context.
    /// </summary>
    /// <remarks>
    /// If the current thread is already executing in the target context, the operation is invoked synchronously.
    /// Otherwise, the operation is enqueued via
    /// <see cref="EnqueueAsync{TResult}(SynchronizationContext, Func{TResult})"/>.
    /// </remarks>
    /// <typeparam name="TResult">
    /// The type of result returned by the operation.
    /// </typeparam>
    /// <param name="context">
    /// The synchronization context on which to run the operation.
    /// </param>
    /// <param name="operation">
    /// The operation to run.
    /// </param>
    /// <returns>
    /// Returns a task that completes with the operation's result when the operation has finished executing.
    /// </returns>
    public static Task<TResult> InvokeAsync<TResult>(this SynchronizationContext context, Func<TResult> operation)
    {
        context.ThrowIfNull();
        operation.ThrowIfNull();

        if (context.HasThreadAccess())
        {
            try
            {
                TResult result = operation();
                return Task.FromResult(result);
            }
            catch (Exception exception)
            {
                return Task.FromException<TResult>(exception);
            }
        }

        return context.EnqueueAsync(operation);
    }

    /// <summary>
    /// Invokes an asynchronous operation on the specified synchronization context, executing synchronously if
    /// already on that context.
    /// </summary>
    /// <remarks>
    /// If the current thread is already executing in the target context, the operation is invoked synchronously.
    /// Otherwise, the operation is enqueued via
    /// <see cref="EnqueueAsync{TResult}(SynchronizationContext, Func{Task{TResult}})"/>.
    /// </remarks>
    /// <typeparam name="TResult">
    /// The type of result returned by the operation.
    /// </typeparam>
    /// <param name="context">
    /// The synchronization context on which to run the operation.
    /// </param>
    /// <param name="operation">
    /// The asynchronous operation to run.
    /// </param>
    /// <returns>
    /// Returns a task that completes with the operation's result when the operation has finished executing.
    /// </returns>
    public static Task<TResult> InvokeAsync<TResult>(this SynchronizationContext context, Func<Task<TResult>> operation)
    {
        context.ThrowIfNull();
        operation.ThrowIfNull();

        if (context.HasThreadAccess())
            return operation();

        return context.EnqueueAsync(operation);
    }

    /// <summary>
    /// Returns an awaitable that, when awaited, continues execution in the specified synchronization context.
    /// </summary>
    /// <remarks>
    /// Awaiting the returned value guarantees that execution will continue within the specified context. If already
    /// executing in that context, execution continues synchronously. Otherwise, the continuation is posted to the
    /// context's queue.
    /// </remarks>
    /// <param name="context">
    /// The synchronization context in which to continue execution. When <c>null</c>, the thread pool context is
    /// used.
    /// </param>
    /// <returns>
    /// Returns an awaitable that continues execution in <paramref name="context"/>.
    /// </returns>
    [SuppressMessage("Usage", "VSTHRD200:Use Async suffix for async methods",
        Justification = "This method returns a custom awaitable type that implements the awaiter pattern.")]
    public static SynchronizationContextAwaiter ContinueInAsync(this SynchronizationContext? context)
    {
        return new SynchronizationContextAwaiter(context ?? new());
    }

    /// <summary>
    /// Indicates whether the current thread has access to the specified synchronization context.
    /// </summary>
    /// <param name="context">
    /// A synchronization context.
    /// </param>
    /// <returns>
    /// Returns <c>true</c> if the current thread is executing in <paramref name="context"/>; otherwise,
    /// <c>false</c>.
    /// </returns>
    public static bool HasThreadAccess(this SynchronizationContext? context)
    {
        return Equals(SynchronizationContext.Current, context)
            || (IsDefaultContext(SynchronizationContext.Current) && IsDefaultContext(context));
    }

    /// <summary>
    /// Indicates whether a synchronization context is the default (thread pool) context.
    /// </summary>
    /// <param name="context">
    /// A synchronization context.
    /// </param>
    /// <returns>
    /// Returns <c>true</c> if <paramref name="context"/> is <c>null</c> or the base
    /// <see cref="SynchronizationContext"/> type; otherwise, <c>false</c>.
    /// </returns>
    public static bool IsDefaultContext(SynchronizationContext? context)
    {
        return context is null || Equals(context.GetType(), typeof(SynchronizationContext));
    }

    #endregion

    #region Internal Static

    internal static void RaiseObservedTaskFaulted(ObservedTaskExceptionEventArgs args)
    {
        ObservedTaskFaulted?.Invoke(null, args);
    }

    #endregion
}
