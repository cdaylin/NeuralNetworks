using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.WeightInitialization;

/// <summary>
/// An initializer for neural network connection weights that generates weights using the normalized Glorot
/// algorithm with a uniform distribution.
/// </summary>
public sealed class GlorotUniformWeightInitializer : IConnectionWeightInitializer
{
    /// <summary>
    /// Constructs a weight initializer.
    /// </summary>
    /// <param name="maxRange">
    /// The maximum range for weights.  (See <see cref="MaxRange"/>.)
    /// </param>
    public GlorotUniformWeightInitializer(float maxRange = 1)
    {
        maxRange.ThrowIfNotPositive();

        MaxRange = maxRange;
    }

    /// <summary>
    /// Abbreviated name for the weight initializer algorithm.
    /// </summary>
    [JsonIgnore]
    public string ShortName => "Glorot";

    /// <summary>
    /// Full name for the weight initializer algorithm.
    /// </summary>
    [JsonIgnore]
    public string LongName => "Normalized Uniform Glorot";

    /// <summary>
    /// Maximum range for weights.
    /// </summary>
    /// <remarks>
    /// For neurons with a small number of connections, the Glorot algorithm can generate weights within an
    /// excessively large range.  This property sets the maximum range allowed. 
    /// </remarks>
    public float MaxRange { get; }

    /// <inheritdoc cref="IConnectionWeightInitializer.GenerateWeight(int, int)"/>
    public float GenerateWeight(int inConnectionCount, int outConnectionCount)
    {
        float range = CalculateRange(inConnectionCount, outConnectionCount);

        return Random.Shared.NextSingle() * range - (range / 2);
    }

    public float CalculateRange(int inConnectionCount, int outConnectionCount)
    {
        return Math.Min(MaxRange, 2 * (float)Math.Sqrt(6.0 / (inConnectionCount + outConnectionCount)));
    }
}
