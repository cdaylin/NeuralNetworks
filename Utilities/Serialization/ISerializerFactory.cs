namespace Daylin.Utilities.Serialization;

/// <summary>
/// Factory for creating serializers and deserializers with named configuration profiles.
/// </summary>
public interface ISerializerFactory
{
    /// <summary>
    /// Creates a serializer configured with the specified profile.
    /// </summary>
    /// <param name="profile">
    /// Configuration profile.
    /// </param>
    /// <returns>
    /// A serializer configured according to the profile.
    /// </returns>
    IObjectSerializer GetSerializer(SerializationProfile profile);

    /// <summary>
    /// Creates a deserializer configured with the specified profile.
    /// </summary>
    /// <param name="profile">
    /// Configuration profile.
    /// </param>
    /// <returns>
    /// A deserializer configured according to the profile.
    /// </returns>
    IObjectDeserializer GetDeserializer(SerializationProfile profile);
}
