namespace Daylin.Utilities.Serialization;

/// <summary>
/// Responsible for deserializing an object graph from a string.
/// </summary>
public interface IObjectDeserializer
{
    /// <summary>
    /// Deserializes an object from a string.
    /// </summary>
    /// <typeparam name="T">
    /// Type of object to deserialize.
    /// </typeparam>
    /// <param name="source">
    /// Serialized representation of an object of type <typeparamref name="T"/>.
    /// </param>
    /// <returns>
    /// An object deserialized from <paramref name="source"/>.
    /// </returns>
    T? Deserialize<T>(string source);
}
