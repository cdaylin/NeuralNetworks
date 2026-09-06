namespace Daylin.Utilities.Extensions;

/// <summary>
/// Extension methods for class <see cref="string"/>.
/// </summary>
public static class StringExtensions
{
    /// <summary>
    /// Determines whether a string is empty.
    /// </summary>
    /// <param name="value">
    /// A string.
    /// </param>
    /// <returns>
    /// Returns <c>true</c> if and only if <paramref name="value"/> is empty.
    /// </returns>
    public static bool IsEmpty(this string value)
    {
        value.ThrowIfNull();

        return value == string.Empty;
    }

    /// <summary>
    /// Determines whether a string is empty or contains only whitespace characters.
    /// </summary>
    /// <param name="value">
    /// A string.
    /// </param>
    /// <returns>
    /// Returns <c>true</c> if and only if <paramref name="value"/> is the empty string or contains only whitespace
    /// characters.
    /// </returns>
    public static bool IsEmptyOrWhitespace(this string value)
    {
        value.ThrowIfNull();

        return string.IsNullOrWhiteSpace(value);
    }

    public static bool StartsWithWhitespace(this string value)
    {
        value.ThrowIfNull();

        return value.TrimStart() != value;
    }

    public static bool EndsWithWhitespace(this string value)
    {
        value.ThrowIfNull();

        return value.TrimEnd() != value;
    }
}
