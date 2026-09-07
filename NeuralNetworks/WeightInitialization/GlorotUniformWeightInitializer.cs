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
    /// <param name="seed">
    /// A seed for the random number generator used to generate weights, or <see langword="null"/> to draw
    /// from a shared generator that is not seeded.  Supplying a seed makes the generated weights
    /// reproducible, so that the training of a network can be repeated exactly.
    /// </param>
    public GlorotUniformWeightInitializer(float maxRange = 1, int? seed = null)
    {
        maxRange.ThrowIfNotPositive();

        MaxRange = maxRange;
        SeededRandom = seed is null ? null : new Random(seed.Value);
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

        return NextSingle() * range - (range / 2);
    }

    public float CalculateRange(int inConnectionCount, int outConnectionCount)
    {
        return Math.Min(MaxRange, 2 * (float)Math.Sqrt(6.0 / (inConnectionCount + outConnectionCount)));
    }

    /// <summary>
    /// A seeded random number generator, or <see langword="null"/> when no seed was supplied.
    /// </summary>
    private Random? SeededRandom { get; }

    /// <summary>
    /// Generates the next random value within the range [0, 1).
    /// </summary>
    /// <returns>
    /// Returns the next random value within the range [0, 1).
    /// </returns>
    private float NextSingle()
    {
        if (SeededRandom is null)
            return Random.Shared.NextSingle();

        // A seeded Random is not thread-safe, and GenerateWeight is required to be.  (See
        // IConnectionWeightInitializer.)
        lock (SeededRandom)
            return SeededRandom.NextSingle();
    }
}
