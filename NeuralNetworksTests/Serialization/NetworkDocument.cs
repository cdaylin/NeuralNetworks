using System.Text.Json;
using System.Text.Json.Nodes;

namespace Daylin.NeuralNetworks.Serialization;

/// <summary>
/// The JSON document a neural network serializes to, with the weights and biases of its neural layers exposed
/// for reading and replacement.
/// </summary>
/// <remarks>
/// Neural layers are internal to the library, and their weights are reachable only through serialization.  A
/// test that needs a network with known weights therefore builds one, replaces the weights in its document,
/// and constructs a network from the result.
/// <para>
/// Reading weights back the same way is what lets a test state the weights a training step should have
/// produced, rather than inferring them from the network's output.
/// </para>
/// </remarks>
internal sealed class NetworkDocument
{
    #region Public Static

    /// <summary>
    /// Creates a document from a neural network.
    /// </summary>
    /// <param name="network">
    /// The network to serialize.
    /// </param>
    /// <returns>
    /// Returns a document holding the serialized network.
    /// </returns>
    public static NetworkDocument FromNetwork(INeuralNetwork network)
    {
        network.ThrowIfNull();

        return new NetworkDocument(JsonSerializer.SerializeToNode(network)!);
    }

    /// <summary>
    /// Creates a document from serialized JSON.
    /// </summary>
    /// <param name="json">
    /// JSON holding a serialized neural network.
    /// </param>
    /// <returns>
    /// Returns a document holding the parsed JSON.
    /// </returns>
    public static NetworkDocument FromJson(string json)
    {
        json.ThrowIfNullOrEmptyOrWhitespace();

        return new NetworkDocument(JsonNode.Parse(json)!);
    }

    #endregion

    #region Private Static

    private static IReadOnlyList<string> NeuralLayerTypeNames { get; } =
        ["DenseLayer", "SparseLayer", "ConvolutionalLayer"];

    private static IEnumerable<JsonObject> FindNeuralLayers(JsonNode node)
    {
        if (node is JsonArray elements)
            return elements.SelectMany(element => FindNeuralLayers(element!));

        if (node is not JsonObject properties)
            return [];

        IEnumerable<JsonObject> nestedLayers = properties
            .Where(property => property.Value is not null)
            .SelectMany(property => FindNeuralLayers(property.Value!));

        return IsNeuralLayer(properties) ? nestedLayers.Prepend(properties) : nestedLayers;
    }

    private static bool IsNeuralLayer(JsonObject properties)
    {
        return properties["$type"] is JsonValue typeName
            && NeuralLayerTypeNames.Contains((string)typeName!);
    }

    #endregion

    #region Construction

    private NetworkDocument(JsonNode root)
    {
        Root = root;

        NeuralLayers = FindNeuralLayers(root)
            .Select(layer => new NeuralLayerDocument(layer))
            .ToList();
    }

    #endregion

    #region Public

    /// <summary>
    /// The network's neural layers, in the order in which they appear in the document.
    /// </summary>
    /// <remarks>
    /// Transformation layers hold no weights and are not listed.
    /// </remarks>
    public IReadOnlyList<NeuralLayerDocument> NeuralLayers { get; }

    /// <summary>
    /// Constructs a neural network from the document.
    /// </summary>
    /// <returns>
    /// Returns a network carrying the document's weights and biases.
    /// </returns>
    public INeuralNetwork BuildNetwork() => JsonSerializer.Deserialize<INeuralNetwork>(ToJson())!;

    public string ToJson() => Root.ToJsonString();

    #endregion

    #region Private

    private JsonNode Root { get; }

    #endregion
}
