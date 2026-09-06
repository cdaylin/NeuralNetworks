namespace Daylin.Utilities.Threading;

/// <summary>
/// Event arguments for the <see cref="SynchronizationContexts.ObservedTaskFaulted"/> event.
/// </summary>
public sealed class ObservedTaskExceptionEventArgs : EventArgs
{
    public ObservedTaskExceptionEventArgs(
        Exception exception,
        string callerFilePath,
        string callerMemberName,
        int callerLineNumber)
    {
        exception.ThrowIfNull();
        callerFilePath.ThrowIfNull();
        callerMemberName.ThrowIfNull();
        callerLineNumber.ThrowIfNotPositive();

        Exception = exception;
        CallerFilePath = callerFilePath;
        CallerMemberName = callerMemberName;
        CallerLineNumber = callerLineNumber;
    }

    /// <summary>
    /// The exception that was thrown by the observed task.
    /// </summary>
    public Exception Exception { get; }

    /// <summary>
    /// Path to the file from which the task was observed.
    /// </summary>
    public string CallerFilePath { get; }

    /// <summary>
    /// Name of the member from which the task was observed.
    /// </summary>
    public string CallerMemberName { get; }

    /// <summary>
    /// Line number from which the task was observed.
    /// </summary>
    public int CallerLineNumber { get; }

    /// <summary>
    /// A brief description of the exception event, not including details from the exception itself.
    /// </summary>
    public string Message
    {
        get
        {
            return Exception is OperationCanceledException
                ? FormatCancellationMessage(CallerFilePath, CallerMemberName, CallerLineNumber)
                : FormatFaultMessage(CallerFilePath, CallerMemberName, CallerLineNumber);
        }
    }

    /// <summary>
    /// Overridden to return text that includes both <see cref="Message"/> and <see cref="Exception.ToString"/>.
    /// </summary>
    public override string ToString()
    {
        return Message + Environment.NewLine + Exception;
    }

    #region Private Static

    private static string FormatCancellationMessage(
        string callerFilePath,
        string callerMemberName,
        int lineNumber)
    {
        string description = "An observed async operation was cancelled.";

        return FormatMessage(description, callerFilePath, callerMemberName, lineNumber);
    }

    private static string FormatFaultMessage(
        string callerFilePath,
        string callerMemberName,
        int callerLineNumber)
    {
        string description = "An exception was thrown from an observed async operation.";

        return FormatMessage(description, callerFilePath, callerMemberName, callerLineNumber);
    }

    private static string FormatMessage(
        string description,
        string callerFilePath,
        string callerMemberName,
        int callerLineNumber)
    {
        return $"{description}  {FormatCallerText(callerFilePath, callerMemberName, callerLineNumber)}";
    }

    private static string FormatCallerText(
        string callerFilePath,
        string callerMemberName,
        int callerLineNumber)
    {
        return $"Called from \"{Path.GetFileName(callerFilePath)}\", {callerMemberName}, line {callerLineNumber}.";
    }

    #endregion
}
