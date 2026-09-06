namespace Daylin.NeuralNetworks.WeightInitialization;

/// <summary>
/// Provides commonly used connection weight initializers.
/// </summary>
public static class WeightInitializers
{
    /// <summary>
    /// Normalized Glorot algorithm with a uniform distribution and a maximum range of (-.5, .5).
    /// </summary>
    public static IConnectionWeightInitializer Glorot { get; } = new GlorotUniformWeightInitializer();

    /// <summary>
    /// He algorithm with a uniform distribution and a maximum range of (-.5, .5).
    /// </summary>
    public static IConnectionWeightInitializer He { get; } = new HeUniformWeightInitializer();
}
