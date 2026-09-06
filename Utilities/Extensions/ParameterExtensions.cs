using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;

namespace Daylin.Utilities.Extensions;

/// <summary>
/// Extension methods for validating method arguments.
/// </summary>
/// <remarks>
/// The extension methods defined on this class operate on parameters of a variety of types. What is common to all of
/// these extension methods is that they are intended to validate method arguments, throwing some type of
/// <see cref="ArgumentException"/> when validation fails.
/// <para>
/// These methods intentionally return <c>void</c> rather than returning the argument being validated. Although this
/// disallows chaining method calls, it (more importantly) prevents ambiguities that arise occassionally when using
/// generic types as return values for extension methods.
/// </para>
/// </remarks>
public static class ParameterExtensions
{
    #region Object

    [StackTraceHidden]
    public static void ThrowIfNull(
        [NotNull] this object? argument,
        [CallerArgumentExpression(nameof(argument))] string? argumentExpression = null,
        [CallerMemberName] string? callerMemberName = null)
    {
        if (argument is not null)
            return;

        string? message = FormatExceptionMessage(
            GetCallerTypeName(),
            callerMemberName,
            argumentExpression,
            "must not be null");

        throw new ArgumentNullException(argumentExpression, message);
    }

    [StackTraceHidden]
    public static void ThrowIfEquals(
        this object? argument,
        object? comparand,
        [CallerArgumentExpression(nameof(argument))] string? argumentExpression = null,
        [CallerArgumentExpression(nameof(comparand))] string? comparandExpression = null,
        [CallerMemberName] string? callerMemberName = null)
    {
        if (!Equals(argument, comparand))
            return;

        string? message = FormatExceptionMessage(
            GetCallerTypeName(),
            callerMemberName,
            argumentExpression,
            $"must not be equal to \"{comparandExpression}\"");

        throw new ArgumentException(argumentExpression, message);
    }

    [StackTraceHidden]
    public static void ThrowIfNotAssignableTo<T>(
        [NotNull] this object? argument,
        [CallerArgumentExpression(nameof(argument))] string? argumentExpression = null,
        [CallerMemberName] string? callerMemberName = null)
    {
        argument.ThrowIfNull(argumentExpression, callerMemberName);

        if (argument!.GetType().IsAssignableTo(typeof(T)))
            return;

        string? message = FormatExceptionMessage(
            GetCallerTypeName(),
            callerMemberName,
            argumentExpression,
            $"must be assignable to type {typeof(T).Name}");

        throw new ArgumentException(argumentExpression, message);
    }

    #endregion

    #region Bool

    [StackTraceHidden]
    public static void ThrowIfFalse(
        this bool argument,
        [CallerArgumentExpression(nameof(argument))] string? argumentExpression = null,
        [CallerMemberName] string? callerMemberName = null)
    {
        if (argument is true)
            return;

        string? message = FormatExceptionMessage(
            GetCallerTypeName(),
            callerMemberName,
            argumentExpression,
            "must be true");

        throw new ArgumentNullException(argumentExpression, message);
    }

    [StackTraceHidden]
    public static void ThrowIfTrue(
        this bool argument,
        [CallerArgumentExpression(nameof(argument))] string? argumentExpression = null,
        [CallerMemberName] string? callerMemberName = null)
    {
        if (argument is false)
            return;

        string? message = FormatExceptionMessage(
            GetCallerTypeName(),
            callerMemberName,
            argumentExpression,
            "must be false");

        throw new ArgumentNullException(argumentExpression, message);
    }

    #endregion

    #region String

    [StackTraceHidden]
    public static void ThrowIfNullOrEmpty(
        [NotNull] this string? argument,
        [CallerArgumentExpression(nameof(argument))] string? argumentExpression = null,
        [CallerMemberName] string? callerMemberName = null)
    {
        argument.ThrowIfNull(argumentExpression, callerMemberName);

        if (argument!.Length != 0)
            return;

        string? message = FormatExceptionMessage(
            GetCallerTypeName(),
            callerMemberName,
            argumentExpression,
            "must not be the empty string");

        throw new ArgumentException(argumentExpression, message);
    }

    /// <summary>
    /// Throws <see cref="ArgumentException"/> if the argument is null or empty or consists only of whitespace
    /// characters.
    /// </summary>
    /// <param name="argument">
    /// Argument to validate.
    /// </param>
    /// <param name="argumentExpression">
    /// (Optional) Description of the argument.
    /// </param>
    /// <param name="callerMemberName">
    /// (Optional) Name of the calling member.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="argument"/> is null or empty or consists only of whitespace characters.
    /// </exception>
    [StackTraceHidden]
    public static void ThrowIfNullOrEmptyOrWhitespace(
        [NotNull] this string? argument,
        [CallerArgumentExpression(nameof(argument))] string? argumentExpression = null,
        [CallerMemberName] string? callerMemberName = null)
    {
        argument.ThrowIfNullOrEmpty(argumentExpression, callerMemberName);

        if (!argument.IsEmptyOrWhitespace())
            return;

        string? message = FormatExceptionMessage(
            GetCallerTypeName(),
            callerMemberName,
            argumentExpression,
            "must not contain only whitespace characters");

        throw new ArgumentException(argumentExpression, message);
    }

    /// <summary>
    /// Throws <see cref="ArgumentException"/> if the argument starts with a whitespace character.
    /// </summary>
    /// <param name="argument">
    /// Argument to validate.
    /// </param>
    /// <param name="argumentExpression">
    /// (Optional) Description of the argument.
    /// </param>
    /// <param name="callerMemberName">
    /// (Optional) Name of the calling member.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="argument"/> starts with a whitespace character.
    /// </exception>
    [StackTraceHidden]
    public static void ThrowIfStartsWithWhitespace(
        this string? argument,
        [CallerArgumentExpression(nameof(argument))] string? argumentExpression = null,
        [CallerMemberName] string? callerMemberName = null)
    {
        if (argument is null)
            return;

        if (!argument.StartsWithWhitespace())
            return;

        string? message = FormatExceptionMessage(
            GetCallerTypeName(),
            callerMemberName,
            argumentExpression,
            "must not start with a whitespace character");

        throw new ArgumentException(argumentExpression, message);
    }

    /// <summary>
    /// Throws <see cref="ArgumentException"/> if the argument ends with a whitespace character.
    /// </summary>
    /// <param name="argument">
    /// Argument to validate.
    /// </param>
    /// <param name="argumentExpression">
    /// (Optional) Description of the argument.
    /// </param>
    /// <param name="callerMemberName">
    /// (Optional) Name of the calling member.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="argument"/> ends with a whitespace character.
    /// </exception>
    [StackTraceHidden]
    public static void ThrowIfEndsWithWhitespace(
        this string? argument,
        [CallerArgumentExpression(nameof(argument))] string? argumentExpression = null,
        [CallerMemberName] string? callerMemberName = null)
    {
        if (argument is null)
            return;

        if (!argument.EndsWithWhitespace())
            return;

        string? message = FormatExceptionMessage(
            GetCallerTypeName(),
            callerMemberName,
            argumentExpression,
            "must not end with a whitespace character");

        throw new ArgumentException(argumentExpression, message);
    }

    #endregion

    #region INumber<TSelf>

    [StackTraceHidden]
    public static void ThrowIfNegative<T>(
        this T argument,
        [CallerArgumentExpression(nameof(argument))] string? argumentExpression = null,
        [CallerMemberName] string? callerMemberName = null)
        where T : INumber<T>
    {
        if (argument >= T.Zero)
            return;

        string? message = FormatExceptionMessage(
            GetCallerTypeName(),
            callerMemberName,
            argumentExpression,
            "must not be negative");

        throw new ArgumentOutOfRangeException(argumentExpression, argument, message);
    }

    [StackTraceHidden]
    public static void ThrowIfNotPositive<T>(
        this T argument,
        [CallerArgumentExpression(nameof(argument))] string? argumentExpression = null,
        [CallerMemberName] string? callerMemberName = null)
        where T : INumber<T>
    {
        if (argument > T.Zero)
            return;

        string? message = FormatExceptionMessage(
            GetCallerTypeName(),
            callerMemberName,
            argumentExpression,
            "must be positive");

        throw new ArgumentOutOfRangeException(argumentExpression, argument, message);
    }

    [StackTraceHidden]
    public static void ThrowIfLessThan<T>(
        this T argument,
        T comparand,
        [CallerArgumentExpression(nameof(argument))] string? argumentExpression = null,
        [CallerArgumentExpression(nameof(comparand))] string? comparandExpression = null,
        [CallerMemberName] string? callerMemberName = null)
        where T : INumber<T>
    {
        if (argument >= comparand)
            return;

        string? message = FormatExceptionMessage(
            GetCallerTypeName(),
            callerMemberName,
            argumentExpression,
            $"must not be less than {comparandExpression}");

        throw new ArgumentOutOfRangeException(argumentExpression, argument, message);
    }

    [StackTraceHidden]
    public static void ThrowIfGreaterThan<T>(
        this T argument,
        T comparand,
        [CallerArgumentExpression(nameof(argument))] string? argumentExpression = null,
        [CallerArgumentExpression(nameof(comparand))] string? comparandExpression = null,
        [CallerMemberName] string? callerMemberName = null)
        where T : INumber<T>
    {
        if (argument <= comparand)
            return;

        string? message = FormatExceptionMessage(
            GetCallerTypeName(),
            callerMemberName,
            argumentExpression,
            $"must not be greater than {comparandExpression}");

        throw new ArgumentOutOfRangeException(argumentExpression, argument, message);
    }

    [StackTraceHidden]
    public static void ThrowIfGreaterThanOrEqualTo<T>(
        this T argument,
        T comparand,
        [CallerArgumentExpression(nameof(argument))] string? argumentExpression = null,
        [CallerArgumentExpression(nameof(comparand))] string? comparandExpression = null,
        [CallerMemberName] string? callerMemberName = null)
        where T : INumber<T>
    {
        if (argument < comparand)
            return;

        string? message = FormatExceptionMessage(
            GetCallerTypeName(),
            callerMemberName,
            argumentExpression,
            $"must be less than {comparandExpression}");

        throw new ArgumentOutOfRangeException(argumentExpression, argument, message);
    }

    [StackTraceHidden]
    public static void ThrowIfNotEqualTo<T>(
        this T argument,
        T comparand,
        [CallerArgumentExpression(nameof(argument))] string? argumentExpression = null,
        [CallerArgumentExpression(nameof(comparand))] string? comparandExpression = null,
        [CallerMemberName] string? callerMemberName = null)
        where T : INumber<T>
    {
        if (argument == comparand)
            return;

        string? message = FormatExceptionMessage(
            GetCallerTypeName(),
            callerMemberName,
            argumentExpression,
            $"must be equal to {comparandExpression}");

        throw new ArgumentOutOfRangeException(argumentExpression, argument, message);
    }

    [StackTraceHidden]
    public static void ThrowIfOutOfRange<T>(
        this T argument,
        T minValue,
        T maxValue,
        [CallerArgumentExpression(nameof(argument))] string? argumentExpression = null,
        [CallerArgumentExpression(nameof(minValue))] string? minValueExpression = null,
        [CallerArgumentExpression(nameof(maxValue))] string? maxValueExpression = null,
        [CallerMemberName] string? callerMemberName = null)
        where T : INumber<T>
    {
        if (argument >= minValue && argument <= maxValue)
            return;

        string? message = FormatExceptionMessage(
            GetCallerTypeName(),
            callerMemberName,
            argumentExpression,
            "must be within the range [{minValueExpression}, {maxValueExpression}]");

        throw new ArgumentOutOfRangeException(argumentExpression, argument, message);
    }

    #endregion

    #region IEnumerable<T>

    [StackTraceHidden]
    public static void ThrowIfEmpty<T>(
        this IEnumerable<T>? argument,
        [CallerArgumentExpression(nameof(argument))] string? argumentExpression = null,
        [CallerMemberName] string? callerMemberName = null)
    {
        if (argument is null)
            return;

        if (argument.Any())
            return;

        string? message = FormatExceptionMessage(
            GetCallerTypeName(),
            callerMemberName,
            argumentExpression,
            "must not be empty");

        throw new ArgumentException(message, argumentExpression);
    }

    [StackTraceHidden]
    public static void ThrowIfContainsNull<T>(
        this IEnumerable<T>? argument,
        [CallerArgumentExpression(nameof(argument))] string? argumentExpression = null,
        [CallerMemberName] string? callerMemberName = null)
    {
        if (argument is null)
            return;

        if (!argument.Any(element => element is null))
            return;

        string? message = FormatExceptionMessage(
            GetCallerTypeName(),
            callerMemberName,
            argumentExpression,
            "must not contain a null value");

        throw new ArgumentException(message, argumentExpression);
    }

    [StackTraceHidden]
    public static void ThrowIfNotDistinct<T>(
        this IEnumerable<T>? argument,
        [CallerArgumentExpression(nameof(argument))] string? argumentExpression = null,
        [CallerMemberName] string? callerMemberName = null)
    {
        if (argument is null)
            return;

        if (argument.AreDistinct())
            return;

        string? message = FormatExceptionMessage(
            GetCallerTypeName(),
            callerMemberName,
            argumentExpression,
            "values must be distinct");

        throw new ArgumentException(message, argumentExpression);
    }

    #endregion

    #region Private

    private static string? GetCallerTypeName()
    {
        int skipFrameCount = 2;

        while (true)
        {
            string? typeName = new StackFrame(skipFrameCount, false).GetMethod()?.ReflectedType?.Name;

            if (Equals(typeName, nameof(ParameterExtensions)))
            {
                skipFrameCount += 1;
                continue;
            }

            return typeName;
        }
    }

    private static string? FormatExceptionMessage(
        string? callerTypeName,
        string? callerMemberName,
        string? argumentExpression,
        string conditionDescription)
    {
        StringBuilder message = new();

        AppendParameterDescription(message, callerTypeName, callerMemberName);

        message.Append($"'{argumentExpression}' {conditionDescription}.");

        return message.ToString();
    }

    private static void AppendParameterDescription(
        StringBuilder messageBuilder,
        string? callerTypeName,
        string? callerMemberName)
    {
        if (callerTypeName is not null && callerMemberName is not null)
        {
            messageBuilder.Append($"'{callerTypeName}");
            messageBuilder.Append($"{(callerMemberName.StartsWith('.') ? "" : ".")}{callerMemberName}' parameter ");
        }
        else if (callerMemberName is not null)
        {
            messageBuilder.Append($"'{(callerMemberName == ".ctor" ? "Constructor" : callerMemberName)}' parameter ");
        }
        else
        {
            messageBuilder.Append("Parameter ");
        }
    }

    #endregion
}
