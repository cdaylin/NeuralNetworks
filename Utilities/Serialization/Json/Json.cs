using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Schema;

namespace Daylin.Utilities.Serialization.Json;

/// <summary>
/// Static convenience class for JSON serialization operations.
/// </summary>
public static class Json
{
    #region Public Static

    /// <summary>
    /// Shared factory instance for creating serializers and deserializers.
    /// </summary>
    public static JsonSerializerFactory Factory { get; } = new();

    /// <summary>
    /// Serializes an object to a JSON string.
    /// </summary>
    /// <typeparam name="T">
    /// Type of object to serialize.
    /// </typeparam>
    /// <param name="value">
    /// Object to serialize.
    /// </param>
    /// <param name="profile">
    /// Configuration profile. Defaults to <see cref="SerializationProfile.Default"/>.
    /// </param>
    /// <returns>
    /// JSON string representation of the object.
    /// </returns>
    public static string Serialize<T>(T value, SerializationProfile profile = SerializationProfile.Default)
    {
        return Factory.GetSerializer(profile).Serialize(value);
    }

    /// <summary>
    /// Deserializes an object from a JSON string.
    /// </summary>
    /// <typeparam name="T">
    /// Type of object to deserialize.
    /// </typeparam>
    /// <param name="json">
    /// JSON string to deserialize.
    /// </param>
    /// <param name="profile">
    /// Configuration profile. Defaults to <see cref="SerializationProfile.Default"/>.
    /// </param>
    /// <returns>
    /// Deserialized object, or null if the JSON represents null.
    /// </returns>
    public static T? Deserialize<T>(string json, SerializationProfile profile = SerializationProfile.Default)
    {
        return Factory.GetDeserializer(profile).Deserialize<T>(json);
    }

    /// <summary>
    /// Creates serialization options for the specified profile.
    /// </summary>
    /// <remarks>
    /// A new instance is returned on each call. Use this when calling APIs that require
    /// <see cref="JsonSerializerOptions"/> directly, such as <see cref="JsonNode"/> extension methods.
    /// </remarks>
    /// <param name="profile">
    /// Configuration profile.
    /// </param>
    /// <returns>
    /// A new <see cref="JsonSerializerOptions"/> instance configured for the profile.
    /// </returns>
    public static JsonSerializerOptions CreateOptions(SerializationProfile profile)
    {
        return Factory.CreateOptions(profile);
    }

    /// <summary>
    /// Serializes an object to a <see cref="JsonNode"/>.
    /// </summary>
    /// <typeparam name="T">
    /// Type of object to serialize.
    /// </typeparam>
    /// <param name="value">
    /// Object to serialize.
    /// </param>
    /// <param name="profile">
    /// Configuration profile. Defaults to <see cref="SerializationProfile.Default"/>.
    /// </param>
    /// <returns>
    /// A <see cref="JsonNode"/> representing the object.
    /// </returns>
    public static JsonNode? SerializeToNode<T>(T value, SerializationProfile profile = SerializationProfile.Default)
    {
        return Factory.SerializeToNode(value, profile);
    }

    /// <summary>
    /// Generates a JSON schema for the specified type.
    /// </summary>
    /// <typeparam name="T">
    /// The type to generate a schema for.
    /// </typeparam>
    /// <param name="exporterOptions">
    /// Options for schema generation.
    /// </param>
    /// <returns>
    /// A <see cref="JsonNode"/> representing the JSON schema.
    /// </returns>
    public static JsonNode GetJsonSchema<T>(JsonSchemaExporterOptions? exporterOptions = null)
    {
        return Factory.GetJsonSchema(typeof(T), exporterOptions);
    }

    /// <summary>
    /// Generates a JSON schema for the specified type.
    /// </summary>
    /// <param name="type">
    /// The type to generate a schema for.
    /// </param>
    /// <param name="exporterOptions">
    /// Options for schema generation.
    /// </param>
    /// <returns>
    /// A <see cref="JsonNode"/> representing the JSON schema.
    /// </returns>
    public static JsonNode GetJsonSchema(Type type, JsonSchemaExporterOptions? exporterOptions = null)
    {
        return Factory.GetJsonSchema(type, exporterOptions);
    }

    #endregion
}
