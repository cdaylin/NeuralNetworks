using System.Text.Json;

namespace Daylin.Utilities.Serialization.Json;

public class JsonObjectDeserializer : IObjectDeserializer
{
    public JsonObjectDeserializer(JsonSerializerOptions? options = null)
    {
        Options = options;
    }

    public virtual T? Deserialize<T>(string source)
    {
        return JsonSerializer.Deserialize<T>(source, Options);
    }

    protected JsonSerializerOptions? Options { get; }
}
