using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;

namespace Daylin.Utilities.Extensions;

/// <summary>
/// Extension methods for validating type parameters.
/// </summary>
/// <remarks>
/// These extension methods are intended to be used for runtime validation of type parameters within constructors of
/// generic types or within generic methods, for those cases in which generic type constraints are insufficient. These
/// extension methods all throw <see cref="InvalidOperationException"/> when validation fails.
/// </remarks>
public static class TypeParameterExtensions
{
    /// <summary>
    /// Throws <see cref="InvalidOperationException"/> when a type argument is a non-nullable value type.
    /// </summary>
    /// <param name="typeArgument">
    /// A generic type argument. Typically this is passed as <c>typeof(T)</c>, where <c>T</c> is the generic parameter
    /// being validated.
    /// </param>
    /// <param name="argumentExpression">
    /// (Optional) Name or description of the type parameter being validated, for use in the exception message. When
    /// the type argument is passed as <c>typeof(T)</c>, the expression is automatically simplified to 'T'. Provide a
    /// custom expression if doing so will make the exception message easier to understand.
    /// </param>
    /// <exception cref="InvalidOperationException">
    /// Thrown when <paramref name="typeArgument"/> is a non-nullable value type.
    /// </exception>
    [StackTraceHidden]
    public static void ThrowIfNotNullable(
        [NotNull] this Type typeArgument,
        [CallerArgumentExpression(nameof(typeArgument))] string? argumentExpression = null)
    {
        typeArgument.ThrowIfNull();

        if (!typeArgument.IsValueType || Nullable.GetUnderlyingType(typeArgument) is not null)
            return;

        throw new InvalidOperationException(
            $"The generic type argument '{FormatArgumentExpression(argumentExpression)}' must be a nullable type. "
                + $"Type '{typeArgument.FullName}' is not nullable.");
    }

    private static string FormatArgumentExpression(string? argumentExpression)
    {
        if (argumentExpression is null)
            return "";

        // remove 'typeof(_)'
        return Regex.Replace(argumentExpression, @"\btypeof\s*\(\s*([^\)]+?)\s*\)", "$1");
    }
}
