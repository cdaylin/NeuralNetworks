namespace Daylin.Utilities.Attributes;

/// <summary>
/// Associates a name (i.e.: a key) with a value.
/// </summary>
/// <remarks>
/// This attribute supports associating arbitrary metadata with application elements.
/// </remarks>
/// <seealso cref="NamedValueAttributeExtensions"/>
[AttributeUsage(AttributeTargets.All, AllowMultiple = true)]
public class NamedValueAttribute<T> : Attribute
{
    public NamedValueAttribute(string name, T value)
    {
        name.ThrowIfNull();

        Name = name;
        Value = value;
    }

    public string Name { get; }

    public T Value { get; }
}

/// <summary>
/// Provides extension methods for obtaining name/value pairs from <see cref="NamedValueAttribute{T}"/> instances.
/// </summary>
public static class NamedValueAttributeExtensions
{
    /// <summary>
    /// Returns all values associated with a name (i.e.: key) via attributes of type 
    /// <see cref="NamedValueAttribute{T}"/> placed on an enum constant.
    /// </summary>
    /// <typeparam name="T">
    /// Type of value associated with <paramref name="name"/>.
    /// </typeparam>
    /// <param name="enumValue">
    /// An enumeration constant.
    /// </param>
    /// <param name="name">
    /// The name of values to return.
    /// </param>
    /// <returns>
    /// Enumerates all values of type <typeparamref name="T"/> associated with <paramref name="name"/> for
    /// <paramref name="enumValue"/>.
    /// </returns>
    public static IEnumerable<T> GetNamedValues<T>(this Enum enumValue, string name)
    {
        enumValue.ThrowIfNull();
        name.ThrowIfNull();

        return enumValue
            .GetAttributes<NamedValueAttribute<T>>()
            .Where(attribute => Equals(attribute.Name, name))
            .Select(attribute => attribute.Value);
    }
}
