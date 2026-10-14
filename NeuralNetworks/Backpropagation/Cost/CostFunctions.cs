namespace Daylin.NeuralNetworks.Backpropagation.Cost;

/// <summary>
/// Provides commonly used cost functions for training a neural network by backpropagation.
/// </summary>
public static class CostFunctions
{
    /// <summary>
    /// The quadratic cost function (i.e.:  mean squared error, MSE).
    /// </summary>
    public static ICostFunction MeanSquaredError { get; } = new MeanSquaredErrorCostFunction();
}
