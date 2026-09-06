namespace Daylin.Utilities.Exceptions;

/// <summary>
/// An exception that indicates that an unsupported case was encountered during an operation.
/// </summary>
/// <remarks>
/// An example of a good use for this exception class is for indicating that an unhandled value was encountered in
/// a switch statement.
/// </remarks>
[Serializable]
public class UnhandledCaseException : Exception
{
    public static string DefaultMessageFormatString { get; } = "Unhandled case:  {0}.";

    public static UnhandledCaseException Create(string caseDescription)
    {
        return new UnhandledCaseException(string.Format(DefaultMessageFormatString, caseDescription));
    }

    public UnhandledCaseException()
    {
    }

    public UnhandledCaseException(Enum unhandledCase)
        : this(string.Format(DefaultMessageFormatString, unhandledCase.ToString()))
    {
    }

    public UnhandledCaseException(Enum unhandledCase, Exception innerException)
        : this(string.Format(DefaultMessageFormatString, unhandledCase.ToString()), innerException)
    {
    }

    public UnhandledCaseException(string message)
        : base(message)
    {
    }

    public UnhandledCaseException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}