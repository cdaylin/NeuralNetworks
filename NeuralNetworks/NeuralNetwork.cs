using Daylin.NeuralNetworks.Backpropagation;
using Daylin.NeuralNetworks.Layers;

using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks;

internal class NeuralNetwork : INeuralNetwork
{
    #region Construction

    [JsonConstructor]
    public NeuralNetwork(INetworkLayer networkLayer)
    {
        networkLayer.ThrowIfNull();

        NetworkLayer = networkLayer;
    }

    #endregion

    #region Public

    public override string ToString() => ShortDescription;

    [JsonIgnore]
    public string ShortDescription => NetworkLayer.ShortDescription;

    [JsonIgnore]
    public int InputCount => NetworkLayer.InputCount;

    [JsonIgnore]
    public int OutputCount => NetworkLayer.OutputCount;

    public void Initialize() => NetworkLayer.Initialize(Enumerable.Repeat(1, OutputCount));

    public async Task<IEnumerable<float>> CalculateOutputAsync(
        IEnumerable<float> input,
        CancellationToken cancellationToken = default)
    {
        return await NetworkLayer.CalculateOutputAsync(input, cancellationToken);
    }

    public async Task TrainAsync(
        BackpropagationSettings settings,
        IReadOnlyList<float> input,
        IReadOnlyList<float> targetOutput,
        CancellationToken cancellationToken = default)
    {
        settings.ThrowIfNull();
        input.Count.ThrowIfNotEqualTo(NetworkLayer.InputCount);
        targetOutput.Count.ThrowIfNotEqualTo(NetworkLayer.OutputCount);

        (IEnumerable<float> output, BackpropagateErrorsAsync backpropagateErrorsAsync) =
            await NetworkLayer.FeedForwardAsync(
                input: input,
                backpropagateErrorsAsync: null,
                cancellationToken: cancellationToken);

        cancellationToken.ThrowIfCancellationRequested();

        IEnumerable<float> errors = CalculateErrors(
            errorFunction: settings.ErrorFunction,
            targetOutput: targetOutput,
            actualOutput: output);

        cancellationToken.ThrowIfCancellationRequested();

        await backpropagateErrorsAsync(
            learningRate: settings.LearningRate,
            errors: errors,
            cancellationToken: cancellationToken);
    }

    #endregion

    #region Private

    [JsonInclude]
    private INetworkLayer NetworkLayer { get; }

    /// <summary>
    /// Calculates the error (i.e.: the partial derivative of the cost function) for each network output value.
    /// </summary>
    private IEnumerable<float> CalculateErrors(
        ErrorFunction errorFunction,
        IEnumerable<float> targetOutput,
        IEnumerable<float> actualOutput)
    {
        return targetOutput
            .Zip(actualOutput)
            .Select(pair => errorFunction(pair.First, pair.Second));
    }

    #endregion
}
