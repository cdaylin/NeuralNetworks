using System.Reflection;

namespace Daylin.Utilities.Extensions;

/// <summary>
/// Extension methods for type <see cref="object"/>.
/// </summary>
public static class ObjectExtensions
{
    /// <summary>
    /// Enumerates a single object.
    /// </summary>
    /// <typeparam name="T">
    /// Type of object to enumerate.
    /// </typeparam>
    /// <param name="item">
    /// Object to enumerate.
    /// </param>
    /// <returns>
    /// Returns an <see cref="IEnumerable{T}"/> that enumerates <paramref name="item"/>.
    /// </returns>
    public static IEnumerable<T> ToEnumerable<T>(this T item)
    {
        yield return item;
    }

    /// <summary>
    /// Gets the value of a public property via reflection.
    /// </summary>
    /// <typeparam name="T">
    /// Return type.  The type must be assignable from the retrieved property's type.
    /// </typeparam>
    /// <param name="item">
    /// An object.
    /// </param>
    /// <param name="propertyName">
    /// The name of a public property.
    /// </param>
    /// <returns>
    /// Returns the value of the specified property from the object.
    /// </returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="item"/> or <paramref name="propertyName"/> is null.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the property does not exist or is not public.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the property's type is not assignable to type <typeparamref name="T"/>.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the property is an indexed property (requiring one or more parameters).
    /// </exception>
    public static T GetPropertyValue<T>(this object item, string propertyName)
    {
        item.ThrowIfNull();
        propertyName.ThrowIfNull();

        PropertyInfo? propertyInfo = item.GetType().GetProperty(propertyName);

        if (propertyInfo is null)
        {
            throw new ArgumentException(
                $"Unable to find a public property named '{propertyName}'"
                    + $" on type '{item.GetType()}'.", nameof(propertyName));
        }

        if (!propertyInfo.PropertyType.IsAssignableTo(typeof(T)))
            throw new ArgumentException($"Property '{propertyName}' is not assignable to type '{typeof(T)}'.");

        if (propertyInfo.GetIndexParameters().Any())
            throw new ArgumentException($"Property '{propertyName}' is an indexed property.");

        object? value = propertyInfo.GetValue(item);

        return (T)value!;
    }
}
