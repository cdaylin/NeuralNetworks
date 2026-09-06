using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.WeightInitialization;

/// <summary>
/// An initializer for neural network connection weights that generates weights using the He algorithm for a
/// uniform distribution.
/// </summary>
public sealed class HeUniformWeightInitializer : IConnectionWeightInitializer
{
    /// <summary>
    /// Constructs a weight initializer.
    /// </summary>
    /// <param name="maxRange">
    /// The maximum range for weights.  (See <see cref="MaxRange"/>.)
    /// </param>
    [JsonConstructor]
    public HeUniformWeightInitializer(float maxRange = 1)
    {
        maxRange.ThrowIfNotPositive();

        MaxRange = maxRange;
    }

    /// <summary>
    /// Abbreviated name for the weight initializer algorithm.
    /// </summary>
    [JsonIgnore]
    public string ShortName => "He";

    /// <summary>
    /// Full name for the weight initializer algorithm.
    /// </summary>
    [JsonIgnore]
    public string LongName => "Uniform He";

    /// <summary>
    /// Maximum range for weights.
    /// </summary>
    /// <remarks>
    /// For neurons with a small number of connections, the He algorithm can generate weights within an
    /// excessively large range.  This property sets the maximum range allowed. 
    /// </remarks>
    public float MaxRange { get; }

    /// <inheritdoc cref="IConnectionWeightInitializer.GenerateWeight(int, int)"/>
    public float GenerateWeight(int inConnectionCount, int outConnectionCount)
    {
        float range = CalculateRange(inConnectionCount);

        return Random.Shared.NextSingle() * range - (range / 2);
    }

    public float CalculateRange(int inConnectionCount)
    {
        return Math.Min(MaxRange, 2 * (float)Math.Sqrt(6.0 / inConnectionCount));
    }
}
