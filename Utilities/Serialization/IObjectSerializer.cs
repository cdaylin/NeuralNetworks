namespace Daylin.Utilities.Serialization;

/// <summary>
/// Responsible for serializing an object graph to a string.
/// </summary>
public interface IObjectSerializer
{
    /// <summary>
    /// Serializes an object to a string.
    /// </summary>
    /// <typeparam name="T">
    /// Type of object to serialize.
    /// </typeparam>
    /// <param name="source">
    /// Object to serialize.
    /// </param>
    /// <returns>
    /// A serialized representation of <paramref name="source"/>.
    /// </returns>
    string Serialize<T>(T? source);
}
