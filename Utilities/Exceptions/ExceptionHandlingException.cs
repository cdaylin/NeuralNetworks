namespace Daylin.Utilities.Exceptions;

/// <summary>
/// An exception class for indicating that an exception has been thrown by an exception handling method.
/// </summary>
[Serializable]
public class ExceptionHandlingException : AggregateException
{
    private static string DefaultMessage { get; } =
        "A message handler threw an exception while handling an exception.  " +
        "The first inner exception was thrown first and passed to the handler.  " +
        "The second inner exception was thrown by the handler.";

    /// <summary>
    /// Constructs a new exception object using a default message.
    /// </summary>
    /// <param name="handledException">
    /// The exception object that was being handled.
    /// </param>
    /// <param name="handlerException">
    /// The exception object that was thrown while handling <paramref name="handledException"/>.
    /// </param>
    public ExceptionHandlingException(Exception handledException, Exception handlerException)
        : this(DefaultMessage, handledException, handlerException)
    {
    }

    /// <summary>
    /// Constructs a new exception object.
    /// </summary>
    /// <param name="message">
    /// A message describing the error.
    /// </param>
    /// <param name="handledException">
    /// The exception object that was being handled.
    /// </param>
    /// <param name="handlerException">
    /// The exception object that was thrown while handling <paramref name="handledException"/>.
    /// </param>
    public ExceptionHandlingException(string message, Exception handledException, Exception handlerException)
        : base(message, [handledException, handlerException])
    {
        HandledException = handledException;
        HandlerException = handlerException;
    }

    /// <summary>
    /// The exception object that was being handled.
    /// </summary>
    public Exception HandledException { get; }

    /// <summary>
    /// The exception object that was thrown while handling <see cref="HandledException"/>.
    /// </summary>
    public Exception HandlerException { get; }
}