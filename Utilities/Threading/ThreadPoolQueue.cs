using Daylin.Utilities.Disposable;

using System.Diagnostics.CodeAnalysis;
using System.Threading.Channels;

namespace Daylin.Utilities.Threading;

/// <summary>
/// Responsible for executing queued operations sequentially in the thread pool.
/// </summary>
public sealed class ThreadPoolQueue : AsyncDisposableObject
{
    #region Construction

    public ThreadPoolQueue()
    {
        UnboundedChannelOptions options = new()
        {
            SingleReader = true,
            AllowSynchronousContinuations = false
        };

        OperationChannel = Channel.CreateUnbounded<Func<Task>>(options);
        ConsumerLoopTask = SynchronizationContexts.ThreadPoolContext.EnqueueAsync(ConsumeAsync);
    }

    #endregion

    #region Public

    public void Enqueue(Func<Task> operation)
    {
        operation.ThrowIfNull();
        ThrowExceptionIfDisposed();

        if (!OperationChannel.Writer.TryWrite(operation))
        {
            // check again in case disposal started after first check
            ThrowExceptionIfDisposed();

            throw new InvalidOperationException("Channel is closed.");
        }
    }

    public void Enqueue(Action operation)
    {
        operation.ThrowIfNull();
        ThrowExceptionIfDisposed();

        Enqueue(() =>
        {
            operation();
            return Task.CompletedTask;
        });
    }

    #endregion

    #region Protected

    [SuppressMessage("Usage", "VSTHRD003:Avoid awaiting foreign Tasks",
        Justification = "The task is known to have been previously started.")]
    protected override async Task ReleaseResourcesAsync()
    {
        OperationChannel.Writer.Complete();

        await SynchronizationContexts.ThreadPoolContext.ContinueInAsync();

        await ConsumerLoopTask;

        await base.ReleaseResourcesAsync();
    }

    #endregion

    #region Private

    private Channel<Func<Task>> OperationChannel { get; }

    private Task ConsumerLoopTask { get; }

    private async Task ConsumeAsync()
    {
        ChannelReader<Func<Task>> reader = OperationChannel.Reader;

        await foreach (Func<Task> operation in reader.ReadAllAsync())
        {
            Task task = SynchronizationContexts.ThreadPoolContext.EnqueueAsync(operation);

            // let exceptions propagate as unobserved
            _ = task.ContinueWith(
                _ => { /* intentionally left blank */ },
                TaskScheduler.Default);
        }
    }

    #endregion
}