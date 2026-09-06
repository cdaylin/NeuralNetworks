namespace Daylin.NeuralNetworks.Activation;

/// <summary>
/// Provides commonly used activation functions (and derivatives).
/// </summary>
public static class ActivationFunctions
{
    /// <summary>
    /// Logistic sigmoid function.
    /// </summary>
    /// <remarks>
    /// Range:  (0, 1).
    /// </remarks>
    public static IActivationFunction Sigmoid { get; } = new SigmoidActivationFunction();

    /// <summary>
    /// Hyperbolic tangent function.
    /// </summary>
    /// <remarks>
    /// Range:  (-1, 1).
    /// </remarks>
    public static IActivationFunction Tanh { get; } = new HyperbolicTangentFunction();

    /// <summary>
    /// Softsign function.
    /// </summary>
    /// <remarks>
    /// Range:  (-1, 1).
    /// </remarks>
    public static IActivationFunction Softsign { get; } = new SoftsignActivationFunction();

    /// <summary>
    /// EANAF (efficient asymmetric nonlinear activation function) function.
    /// </summary>
    /// <remarks>
    /// Range:  (~-.25, +inf)
    /// </remarks>
    public static IActivationFunction Eanaf { get; } = new EanafFunction();

    /// <summary>
    /// Rectified linear unit (ReLU) function.
    /// </summary>
    /// <remarks>
    /// Range:  (0, +inf)
    /// </remarks>
    public static IActivationFunction Relu { get; } = new ReluActivationFunction();
}
