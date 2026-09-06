using System.Text.Json;

namespace Daylin.Utilities.Serialization.Json;

public class JsonObjectSerializer : IObjectSerializer
{
    public JsonObjectSerializer(JsonSerializerOptions? options = null)
    {
        Options = options;
    }

    public virtual string Serialize<T>(T? source)
    {
        return JsonSerializer.Serialize(source, Options);
    }

    protected JsonSerializerOptions? Options { get; }
}
