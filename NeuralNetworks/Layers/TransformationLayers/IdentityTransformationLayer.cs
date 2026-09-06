using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Layers.TransformationLayers;

/// <summary>
/// A neural network layer that simply forwards all input values as output values.
/// </summary>
/// <remarks>
/// An <see cref="IdentityTransformationLayer"/> object acts as a "pass-through layer", forwarding each input
/// value to a corresponding output value.
/// <para>
/// One use for an <see cref="IdentityTransformationLayer"/> object is to act as an "input layer".  Another use
/// is to act as a "bypass" to allow a subset of output from a layer to be sent through a subsequent network
/// layer, by concatenating the subsequent layer with an identity transformation layer.
/// </para>
/// </remarks>
internal sealed class IdentityTransformationLayer : INetworkLayer
{
    [JsonConstructor]
    public IdentityTransformationLayer(int inputCount, string layerDescription)
    {
        inputCount.ThrowIfNotPositive();
        layerDescription.ThrowIfNullOrEmptyOrWhitespace();

        InputCount = inputCount;
        LayerDescription = layerDescription;
    }

    public override string ToString() => ShortDescription;

    [JsonIgnore]
    public string ShortDescription => $"({LayerDescription} = {InputCount})";

    /// <summary>
    /// Number of values input to this layer.
    /// </summary>
    public int InputCount { get; }

    /// <summary>
    /// Number of values output from this layer.
    /// </summary>
    [JsonIgnore]
    public int OutputCount => InputCount;

    public IEnumerable<int> Initialize(IEnumerable<int> outputConnectionCounts)
    {
        // nothing to initialize

        return outputConnectionCounts;
    }

    public Task<IEnumerable<float>> CalculateOutputAsync(
        IEnumerable<float> input,
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(input);
    }

    public Task<(IEnumerable<float>, BackpropagateErrorsAsync)> FeedForwardAsync(
        IEnumerable<float> input,
        BackpropagateErrorsAsync? backpropagateErrorsAsync,
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult<(IEnumerable<float>, BackpropagateErrorsAsync)>((input, BackpropagateAsync));

        async Task BackpropagateAsync(
            float learningRate,
            IEnumerable<float> errors,
            CancellationToken cancellationToken)
        {
            if (backpropagateErrorsAsync is null)
                return;

            await backpropagateErrorsAsync(learningRate, errors, cancellationToken);
        }
    }

    [JsonInclude]
    private string LayerDescription { get; }
}
