using System.Runtime.CompilerServices;
using System.Text;

namespace Daylin.Utilities.Disposable;

/// <summary>
/// Responsible for formatting error messages for <see cref="ObjectDisposedException"/>.
/// </summary>
public static class ObjectDisposedExceptionMessageFormatter
{
    public static string FormatExceptionMessage(
        Type disposedType,
        [CallerFilePath] string callerFilePath = "",
        [CallerMemberName] string callerMemberName = "",
        [CallerLineNumber] int callerLineNumber = 0)
    {
        StringBuilder message = new();

        message.Append("An attempt is being made to use an object of type ");
        message.Append(disposedType.Name);
        message.Append(" after the object has been disposed.");

        if (!string.IsNullOrEmpty(callerFilePath))
        {
            message.Append(
                $"  File: \"{callerFilePath}\".  Line: {callerLineNumber}.  Member: \"{callerMemberName}\".");
        }

        return message.ToString();
    }
}
