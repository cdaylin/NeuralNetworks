using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

using Daylin.Utilities.Threading;

namespace Daylin.Utilities.Extensions;

/// <summary>
/// Extension methods for class <see cref="Task"/> and related types.
/// </summary>
[SuppressMessage("Usage", "VSTHRD003:Avoid awaiting foreign Tasks",
    Justification = "This class intentionally awaits foreign Tasks.")]
public static class TaskExtensions
{
    /// <summary>
    /// Observes a task without awaiting it, routing any exceptions to the
    /// <see cref="SynchronizationContexts.ObservedTaskFaulted"/> event.
    /// </summary>
    /// <remarks>
    /// Use this method when a task's result is not needed but exceptions should not go unobserved. Exceptions of
    /// type <see cref="OperationCanceledException"/> are ignored. All other exceptions are reported via the
    /// <see cref="SynchronizationContexts.ObservedTaskFaulted"/> event.
    /// <para>
    /// This method captures information about the caller for use in exception handling and logging.
    /// </para>
    /// </remarks>
    /// <param name="task">
    /// The task to observe.
    /// </param>
    /// <param name="callerFilePath">
    /// Path to the file from which this method was called.
    /// </param>
    /// <param name="callerMemberName">
    /// Name of the member from which this method was called.
    /// </param>
    /// <param name="callerLineNumber">
    /// Line number from which this method was called.
    /// </param>
    [SuppressMessage("Usage", "VSTHRD110:Observe the awaitable result of this method call",
        Justification = "The continuation task is intentionally not awaited; its purpose is to observe the original task.")]
    public static void Observe(
        this Task task,
        [CallerFilePath] string callerFilePath = "",
        [CallerMemberName] string callerMemberName = "",
        [CallerLineNumber] int callerLineNumber = 0)
    {
        task.ThrowIfNull();

        // Detach the continuation from any active trace scope before ContinueWith captures
        // the ExecutionContext. A fire-and-forget fault must not be reported against a scope
        // that may have been disposed by the time the continuation runs (the captured EC
        // preserves a reference to the Activity even after it has been stopped). We clear
        // and restore the caller's Activity.Current so the captured snapshot has no scope,
        // and the calling code observes no change.
        Activity? savedActivity = Activity.Current;
        Activity.Current = null;

        try
        {
            _ = task.ContinueWith(
                completedTask =>
                {
                    if (completedTask.IsFaulted)
                    {
                        Exception exception = completedTask.Exception!.InnerExceptions.Count == 1
                            ? completedTask.Exception.InnerExceptions[0]
                            : completedTask.Exception;

                        if (exception is not OperationCanceledException)
                        {
                            ObservedTaskExceptionEventArgs args = new(
                                exception,
                                callerFilePath,
                                callerMemberName,
                                callerLineNumber);

                            SynchronizationContexts.RaiseObservedTaskFaulted(args);
                        }
                    }
                },
                CancellationToken.None,
                TaskContinuationOptions.OnlyOnFaulted,
                TaskScheduler.Default);
        }
        finally
        {
            Activity.Current = savedActivity;
        }
    }

    /// <summary>
    /// Observes a task without awaiting it, routing any exceptions to the
    /// <see cref="SynchronizationContexts.ObservedTaskFaulted"/> event.
    /// </summary>
    /// <remarks>
    /// Use this method when a task's result is not needed but exceptions should not go unobserved. Exceptions of
    /// type <see cref="OperationCanceledException"/> are ignored. All other exceptions are reported via the
    /// <see cref="SynchronizationContexts.ObservedTaskFaulted"/> event.
    /// <para>
    /// This method captures information about the caller for use in exception handling and logging.
    /// </para>
    /// </remarks>
    /// <typeparam name="TResult">
    /// The type of result returned by the task.
    /// </typeparam>
    /// <param name="task">
    /// The task to observe.
    /// </param>
    /// <param name="callerFilePath">
    /// Path to the file from which this method was called.
    /// </param>
    /// <param name="callerMemberName">
    /// Name of the member from which this method was called.
    /// </param>
    /// <param name="callerLineNumber">
    /// Line number from which this method was called.
    /// </param>
    public static void Observe<TResult>(
        this Task<TResult> task,
        [CallerFilePath] string callerFilePath = "",
        [CallerMemberName] string callerMemberName = "",
        [CallerLineNumber] int callerLineNumber = 0)
    {
        ((Task)task).Observe(callerFilePath, callerMemberName, callerLineNumber);
    }

    /// <summary>
    /// Synchronously awaits completion of a task.
    /// </summary>
    /// <remarks>
    /// This method behaves like <see cref="Task.Wait()"/>, except that it throws exceptions in the same manner
    /// as the <c>await</c> keyword.
    /// <para>
    /// When a task is cancelled or faults, <see cref="Task.Wait()"/> throws <see cref="AggregateException"/>.
    /// This method throws <see cref="OperationCanceledException"/> in the case of cancellation or the original
    /// exception in the case of a fault.
    /// </para>
    /// </remarks>
    /// <param name="task">
    /// A task.
    /// </param>
    /// <exception cref="OperationCanceledException">
    /// Thrown when the task is cancelled.
    /// </exception>
    /// <exception cref="Exception">
    /// Thrown when the task is faulted.
    /// </exception>
    [SuppressMessage("Usage", "VSTHRD002:Avoid problematic synchronous waits",
        Justification = "This method explicitly performs a synchronous wait.")]
    public static void AwaitSync(this Task task)
    {
        task.ThrowIfNull();

        task.GetAwaiter().GetResult();
    }

    /// <summary>
    /// Synchronously awaits completion of a task and returns its result.
    /// </summary>
    /// <remarks>
    /// This method behaves like <see cref="Task.Wait()"/>, except that it throws exceptions in the same manner
    /// as the <c>await</c> keyword.
    /// <para>
    /// When a task is cancelled or faults, <see cref="Task.Wait()"/> throws <see cref="AggregateException"/>.
    /// This method throws <see cref="OperationCanceledException"/> in the case of cancellation or the original
    /// exception in the case of a fault.
    /// </para>
    /// </remarks>
    /// <typeparam name="TResult">
    /// Type of result returned by the task.
    /// </typeparam>
    /// <param name="task">
    /// A task.
    /// </param>
    /// <returns>
    /// Returns the result from parameter <paramref name="task"/>.
    /// </returns>
    /// <exception cref="OperationCanceledException">
    /// Thrown when the task is cancelled.
    /// </exception>
    /// <exception cref="Exception">
    /// Thrown when the task is faulted.
    /// </exception>
    [SuppressMessage("Usage", "VSTHRD002:Avoid problematic synchronous waits",
        Justification = "This method explicitly performs a synchronous wait.")]
    public static TResult AwaitSync<TResult>(this Task<TResult> task)
    {
        task.ThrowIfNull();

        return task.GetAwaiter().GetResult();
    }

    /// <summary>
    /// Ensures that if a task faults, it always faults with an exception of type <see cref="AggregateException"/>.
    /// </summary>
    /// <remarks>
    /// If <paramref name="task"/> faults with an exception of a type other than <see cref="AggregateException"/>, the
    /// original exception is wrapped in a new <see cref="AggregateException"/> object.
    /// <para>
    /// This method can make exception handling simpler for composite tasks:  those tasks that use
    /// <see cref="AggregateException"/> to wrap multiple exceptions. This method guarantees that a faulted task
    /// always throws <see cref="AggregateException"/>, even when the aggregate exception contains only a single
    /// wrapped exception. The caller can simply catch <see cref="AggregateException"/> and enumerate the wrapped
    /// exceptions.
    /// </para>
    /// </remarks>
    /// <param name="task">
    /// A task.
    /// </param>
    /// <returns>
    /// Returns a task that is identical to <paramref name="task"/> except that when it faults it always does
    /// so with an exception of type <see cref="AggregateException"/>.
    /// </returns>
    public static async Task WithAggregateExceptionAsync(this Task task)
    {
        task.ThrowIfNull();

        try
        {
            await task;
        }
        catch (OperationCanceledException) when (task.IsCanceled)
        {
            throw;
        }
        catch (AggregateException)
        {
            throw;
        }
        catch
        {
            throw task.Exception!;
        }
    }

    /// <summary>
    /// Ensures that if a task faults, it always faults with an exception of type <see cref="AggregateException"/>.
    /// </summary>
    /// <remarks>
    /// If <paramref name="task"/> faults with an exception of a type other than <see cref="AggregateException"/>, the
    /// original exception is wrapped in a new <see cref="AggregateException"/> object.
    /// <para>
    /// This method can make exception handling simpler for composite tasks:  those tasks that use
    /// <see cref="AggregateException"/> to wrap multiple exceptions. This method guarantees that a faulted task
    /// always throws <see cref="AggregateException"/>, even when the aggregate exception contains only a single
    /// wrapped exception. The caller can simply catch <see cref="AggregateException"/> and enumerate the wrapped
    /// exceptions.
    /// </para>
    /// </remarks>
    /// <typeparam name="TResult">
    /// Task result type.
    /// </typeparam>
    /// <param name="task">
    /// A task.
    /// </param>
    /// <returns>
    /// Returns a task that is identical to <paramref name="task"/> except that when it faults it always does
    /// so with an exception of type <see cref="AggregateException"/>.
    /// </returns>
    [SuppressMessage("Usage", "VSTHRD103:Call async methods when in an async method",
        Justification = "This method intentionally calls Task.Result, but it will not block.")]
    public static async Task<TResult> WithAggregateExceptionAsync<TResult>(this Task<TResult> task)
    {
        task.ThrowIfNull();

        try
        {
            return await task;
        }
        catch (OperationCanceledException) when (task.IsCanceled)
        {
            throw;
        }
        catch (AggregateException)
        {
            throw;
        }
        catch
        {
            return task.Result; // throws AggregateExeption
        }
    }

    /// <summary>
    /// Catches and ignores <see cref="OperationCanceledException"/>.
    /// </summary>
    /// <remarks>
    /// The task returned by this method will run to completion if the  <paramref name="task"/> parameter either
    /// runs to completion or is cancelled. This method will never throw <see cref="OperationCanceledException"/>.
    /// </remarks>
    /// <param name="task">
    /// A task.
    /// </param>
    /// <returns>
    /// Returns a task that can be used to await this operation.
    /// </returns>
    [SuppressMessage("Usage", "VSTHRD003:Avoid awaiting foreign Tasks",
        Justification = "This method intentionally awaits a foreign Task.")]
    public static async Task IgnoreCancellationAsync(this Task task)
    {
        try
        {
            await task;
        }
        catch (OperationCanceledException)
        {
            // ignore
        }
    }
}
