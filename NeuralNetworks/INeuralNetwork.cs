using Daylin.NeuralNetworks.Backpropagation;

using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks;

[JsonPolymorphic]
[JsonDerivedType(typeof(NeuralNetwork), nameof(NeuralNetwork))]
public interface INeuralNetwork
{
    string ShortDescription { get; }

    /// <summary>
    /// Number of values fed into the network.
    /// </summary>
    int InputCount { get; }

    /// <summary>
    /// Number of values output from the network.
    /// </summary>
    int OutputCount { get; }

    Task<IEnumerable<float>> CalculateOutputAsync(
        IEnumerable<float> input,
        CancellationToken cancellationToken = default);

    Task TrainAsync(
        BackpropagationSettings settings,
        IReadOnlyList<float> input,
        IReadOnlyList<float> targetOutput,
        CancellationToken cancellationToken = default);
}
