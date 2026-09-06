using System.Reflection;
using System.Runtime.ExceptionServices;

namespace Daylin.Utilities.Extensions;

/// <summary>
/// Extension methods for class <see cref="Exception"/>.
/// </summary>
public static class ExceptionExtensions
{
    #region Public

    /// <summary>
    /// Rethrows a previously caught exception without modifying the original stack trace.
    /// </summary>
    /// <param name="exception">
    /// An exception.
    /// </param>
    public static void Rethrow(Exception exception)
    {
        exception.ThrowIfNull();

        ExceptionDispatchInfo.Capture(exception).Throw();
    }

    /// <summary>
    /// Unwraps an exception to get the underlying meaningful exception.
    /// </summary>
    /// <remarks>
    /// This method unwraps wrapper exceptions like <see cref="TargetInvocationException"/> and
    /// <see cref="AggregateException"/> (when it contains a single inner exception) to return the actual exception
    /// that caused the failure. This is useful for logging and error messages, as wrapper exceptions often have
    /// generic messages that obscure the real cause.
    /// </remarks>
    /// <param name="exception">
    /// An exception.
    /// </param>
    /// <returns>
    /// The underlying meaningful exception, or the original exception if it is not a wrapper.
    /// </returns>
    public static Exception Unwrap(this Exception exception)
    {
        exception.ThrowIfNull();

        while (true)
        {
            if (exception is TargetInvocationException { InnerException: { } innerException })
            {
                exception = innerException;
                continue;
            }

            if (exception is AggregateException { InnerExceptions: [{ } singleInnerException] })
            {
                exception = singleInnerException;
                continue;
            }

            return exception;
        }
    }

    #endregion
}
