using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.WeightInitialization;

/// <summary>
/// Responsible for initializing weights of a neuron's incoming connections.
/// </summary>
/// <remarks>
/// A single <see cref="IConnectionWeightInitializer"/> object may be used to initialize weights for multiple
/// neurons, and those initializations may be performed concurrently.  Therefore, <see cref="GenerateWeight"/>
/// must be implemented to be thread-safe.
/// </remarks>
[JsonPolymorphic]
[JsonDerivedType(typeof(GlorotUniformWeightInitializer), typeDiscriminator: "GlorotUniform")]
[JsonDerivedType(typeof(HeUniformWeightInitializer), typeDiscriminator: "HeUniformWeightInitializer")]
public interface IConnectionWeightInitializer
{
    /// <summary>
    /// Abbreviated name for the weight initializer algorithm.
    /// </summary>
    string ShortName { get; }

    /// <summary>
    /// Full name for the weight initializer algorithm.
    /// </summary>
    string LongName { get; }

    /// <summary>
    /// Generates a value for initializing the weight of an incoming connection to a neuron.
    /// </summary>
    /// <param name="inConnectionCount">
    /// Number of incoming connections to the neuron.  (Aka: "fan-in".)
    /// </param>
    /// <param name="outConnectionCount">
    /// Number of outgoing connections from the neuron.  (Aka: "fan-out".)
    /// </param>
    /// <returns>
    /// Returns a value for initializing the weight of an incoming connection to a neuron.
    /// </returns>
    float GenerateWeight(int inConnectionCount, int outConnectionCount);
}
