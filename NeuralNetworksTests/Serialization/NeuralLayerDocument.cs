using System.Text.Json.Nodes;

namespace Daylin.NeuralNetworks.Serialization;

/// <summary>
/// One neural layer within a <see cref="NetworkDocument"/>, with its weights and biases exposed for reading and
/// replacement.
/// </summary>
/// <remarks>
/// A convolutional layer shares one bias and one set of connection weights among all of its output values.  Its
/// document presents them as a bias vector holding one value and a weight matrix holding one row, so that every
/// neural layer is read and written the same way.
/// </remarks>
internal sealed class NeuralLayerDocument
{
    #region Private Static

    private static JsonArray CreateArray(IEnumerable<float> values)
    {
        return new JsonArray(values.Select(value => (JsonNode?)JsonValue.Create(value)).ToArray());
    }

    #endregion

    #region Construction

    public NeuralLayerDocument(JsonObject properties)
    {
        properties.ThrowIfNull();

        Properties = properties;
    }

    #endregion

    #region Public

    public override string ToString() => LayerTypeName;

    /// <summary>
    /// Name of the layer's type, as the JSON type discriminator gives it.
    /// </summary>
    public string LayerTypeName => (string)Properties["$type"]!;

    /// <summary>
    /// The layer's bias for each of its output values.
    /// </summary>
    public IReadOnlyList<float> BiasVector =>
        IsConvolutional ? [(float)Properties["Bias"]!] : ReadVector(Properties["BiasVector"]!);

    /// <summary>
    /// The layer's connection weights, as one row for each of its output values.
    /// </summary>
    /// <remarks>
    /// A row holds one weight for each input value connected to the output value, so a sparse layer's rows are
    /// shorter than its input count wherever the connection mask masks a connection.
    /// </remarks>
    public IReadOnlyList<IReadOnlyList<float>> WeightMatrix =>
        IsConvolutional ? [ReadVector(Properties["WeightVector"]!)] : ReadMatrix(Properties["WeightMatrix"]!);

    /// <summary>
    /// Seed recorded for the layer's weight initializer, or <see langword="null"/> where the document holds
    /// none.
    /// </summary>
    public int? WeightInitializerSeed => (int?)Properties["WeightInitializer"]?["Seed"];

    /// <summary>
    /// Replaces the layer's biases.
    /// </summary>
    /// <param name="biasVector">
    /// One bias for each of the layer's output values.
    /// </param>
    public void SetBiasVector(params IEnumerable<float> biasVector)
    {
        IReadOnlyList<float> biases = biasVector.ToList();

        biases.Count.ThrowIfNotEqualTo(BiasVector.Count);

        if (IsConvolutional)
            Properties["Bias"] = JsonValue.Create(biases.Single());
        else
            Properties["BiasVector"] = CreateArray(biases);
    }

    /// <summary>
    /// Replaces the layer's connection weights.
    /// </summary>
    /// <param name="weightMatrix">
    /// One row of weights for each of the layer's output values, each row holding as many weights as the row it
    /// replaces.
    /// </param>
    public void SetWeightMatrix(params IEnumerable<IEnumerable<float>> weightMatrix)
    {
        IReadOnlyList<IReadOnlyList<float>> rows = weightMatrix
            .Select(row => (IReadOnlyList<float>)row.ToList())
            .ToList();

        ValidateShape(rows);

        if (IsConvolutional)
            Properties["WeightVector"] = CreateArray(rows.Single());
        else
            Properties["WeightMatrix"] = new JsonArray(rows.Select(row => (JsonNode?)CreateArray(row)).ToArray());
    }

    #endregion

    #region Private

    private JsonObject Properties { get; }

    private bool IsConvolutional => LayerTypeName == "ConvolutionalLayer";

    private IReadOnlyList<float> ReadVector(JsonNode values) =>
        ((JsonArray)values).Select(value => (float)value!).ToList();

    private IReadOnlyList<IReadOnlyList<float>> ReadMatrix(JsonNode rows) =>
        ((JsonArray)rows).Select(row => ReadVector(row!)).ToList();

    private void ValidateShape(IReadOnlyList<IReadOnlyList<float>> rows)
    {
        IReadOnlyList<IReadOnlyList<float>> existingRows = WeightMatrix;

        rows.Count.ThrowIfNotEqualTo(existingRows.Count);

        for (int rowIndex = 0; rowIndex < rows.Count; rowIndex++)
            rows[rowIndex].Count.ThrowIfNotEqualTo(existingRows[rowIndex].Count);
    }

    #endregion
}
