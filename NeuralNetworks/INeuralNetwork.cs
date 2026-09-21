using Daylin.NeuralNetworks.Backpropagation;

using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks;

/// <summary>
/// A neural network:  a function from a fixed number of input values to a fixed number of output values, whose
/// weights can be moved by training.
/// </summary>
/// <remarks>
/// A network carries what it needs to calculate output, and nothing about the course of training that produced
/// its weights.  All training -- one sample, a batch, an epoch -- is a <see cref="TrainingSession"/>'s, which
/// holds the network it trains and can be saved and resumed separately from it.
/// </remarks>
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
}
