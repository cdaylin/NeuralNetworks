using Daylin.Utilities.Collections;

using System.Globalization;

namespace Daylin.Utilities.Extensions;

/// <summary>
/// Provides extension methods for working with metadata on objects that implement <see cref="IHasMetadata"/>.
/// </summary>
public static class HasMetadataExtensions
{
    public static T? GetMetadata<T>(this IHasMetadata source, string key, T? defaultValue = default)
    {
        source.ThrowIfNull();
        key.ThrowIfNull();

        if (!source.Metadata.TryGetValue(key, out string? text))
            return defaultValue;

        try
        {
            object? converted = Convert.ChangeType(text, typeof(T), CultureInfo.InvariantCulture);
            return (T?)converted;
        }
        catch (Exception exception)
        {
            throw new FormatException(
                $"Metadata value for key '{key}' could not be converted to {typeof(T).Name}.",
                exception);
        }
    }

    public static void SetMetadata<T>(this IHasMetadata source, string key, T value)
    {
        source.ThrowIfNull();
        key.ThrowIfNull();

        string? text = Convert.ToString(value, CultureInfo.InvariantCulture);
        source.Metadata[key] = text ?? string.Empty;
    }

    public static bool TryGetMetadata<T>(this IHasMetadata source, string key, out T value)
    {
        source.ThrowIfNull();
        key.ThrowIfNull();

        if (source.Metadata.TryGetValue(key, out string? text))
        {
            object? converted = Convert.ChangeType(text, typeof(T), CultureInfo.InvariantCulture);

            if (converted is T typed)
            {
                value = typed;
                return true;
            }
        }

        value = default!;
        return false;
    }

    public static void RemoveMetadata(this IHasMetadata source, string key)
    {
        source.ThrowIfNull();
        key.ThrowIfNull();

        source.Metadata.Remove(key);
    }
}