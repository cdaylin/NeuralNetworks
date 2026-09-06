using Daylin.Utilities.Disposable;

using Microsoft.VisualStudio.Threading;

namespace Daylin.Utilities.Threading;

/// <summary>
/// Hosts a <see cref="SingleThreadedSynchronizationContext"/> that executes operations on a dedicated thread.
/// </summary>
/// <remarks>
/// A <see cref="SingleThreadedSynchronizationContextHost"/> object starts a dedicated thread and sets its 
/// synchronization context to a <see cref="SingleThreadedSynchronizationContext"/> object.  The synchronization
/// context runs a loop that executes operations in the sequence in which they are added to the queue.
/// <para>
/// Disposing a host will stop the processing loop on the associated thread.  If the thread is currently processing an
/// operation, the loop will stop after that operation has completed.
/// </para>
/// </remarks>
public class SingleThreadedSynchronizationContextHost : DisposableObject
{
    public SingleThreadedSynchronizationContextHost(
        string? threadName = null,
        bool isBackgroundThread = true)
    {
        SynchronizationThread = new Thread(Run)
        {
            IsBackground = isBackgroundThread
        };

        if (threadName is not null)
            SynchronizationThread.Name = threadName;

        SynchronizationThread.Start();
    }

    protected override void ReleaseResources()
    {
        Frame!.Continue = false;

        base.ReleaseResources();
    }

    private void Run()
    {
        SingleThreadedSynchronizationContext context = new();

        Frame = new SingleThreadedSynchronizationContext.Frame();

        ContextInitializationTaskCompletionSource.SetResult(context);

        context.PushFrame(Frame);
    }

    public SynchronizationContext Context => ContextInitializationTaskCompletionSource.Task.AwaitSync();

    private TaskCompletionSource<SynchronizationContext> ContextInitializationTaskCompletionSource { get; } = new();

    private SingleThreadedSynchronizationContext.Frame? Frame { get; set; }

    private Thread SynchronizationThread { get; }
}