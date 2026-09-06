using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Schema;

namespace Daylin.Utilities.Serialization.Json;

/// <summary>
/// Factory for creating JSON serializers and deserializers with named configuration profiles.
/// </summary>
public class JsonSerializerFactory : ISerializerFactory
{
    #region Public

    /// <inheritdoc/>
    public IObjectSerializer GetSerializer(SerializationProfile profile)
    {
        return new JsonObjectSerializer(CreateOptions(profile));
    }

    /// <inheritdoc/>
    public IObjectDeserializer GetDeserializer(SerializationProfile profile)
    {
        return new JsonObjectDeserializer(CreateOptions(profile));
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
    public JsonSerializerOptions CreateOptions(SerializationProfile profile)
    {
        return profile switch
        {
            SerializationProfile.Default => new JsonSerializerOptions(),
            SerializationProfile.Persistence => new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                WriteIndented = true
            },
            SerializationProfile.DataTransfer => new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            },
            _ => throw new ArgumentOutOfRangeException(nameof(profile), profile, "Unknown serialization profile.")
        };
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
    /// Configuration profile.
    /// </param>
    /// <returns>
    /// A <see cref="JsonNode"/> representing the object.
    /// </returns>
    public JsonNode? SerializeToNode<T>(T value, SerializationProfile profile = SerializationProfile.Default)
    {
        return JsonSerializer.SerializeToNode(value, CreateOptions(profile));
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
    public JsonNode GetJsonSchema(Type type, JsonSchemaExporterOptions? exporterOptions = null)
    {
        return JsonSerializerOptions.Default.GetJsonSchemaAsNode(type, exporterOptions ?? new());
    }

    #endregion
}
