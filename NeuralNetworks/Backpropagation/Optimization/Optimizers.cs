namespace Daylin.NeuralNetworks.Backpropagation.Optimization;

/// <summary>
/// Provides commonly used optimizers for training a neural network.
/// </summary>
public static class Optimizers
{
    /// <summary>
    /// An optimizer that moves each parameter against its gradient, scaled by the learning rate.
    /// </summary>
    /// <remarks>
    /// One instance serves every caller:  plain gradient descent holds no hyperparameters and keeps no state,
    /// as the activation functions and weight initializers here do.
    /// </remarks>
    public static IOptimizer GradientDescent { get; } = new GradientDescentOptimizer();

    /// <summary>
    /// Constructs an optimizer that accumulates a velocity for each parameter.
    /// </summary>
    /// <param name="momentum">
    /// The fraction of a parameter's velocity that survives a step, in the range [0, 1).
    /// </param>
    /// <returns>
    /// Returns a new optimizer performing gradient descent with momentum.
    /// </returns>
    public static IOptimizer Momentum(float momentum = 0.9f) => new MomentumOptimizer(momentum);

    /// <summary>
    /// Constructs an optimizer that scales each parameter's step by how consistent and how large its recent
    /// gradients have been.
    /// </summary>
    /// <param name="firstMomentDecay">
    /// The fraction of the mean gradient that survives a step, in the range [0, 1).
    /// </param>
    /// <param name="secondMomentDecay">
    /// The fraction of the mean squared gradient that survives a step, in the range [0, 1).
    /// </param>
    /// <param name="epsilon">
    /// A small positive value added to the denominator of a step.
    /// </param>
    /// <returns>
    /// Returns a new optimizer performing adaptive moment estimation.
    /// </returns>
    public static IOptimizer Adam(
        float firstMomentDecay = 0.9f,
        float secondMomentDecay = 0.999f,
        float epsilon = 1e-8f) =>
        new AdamOptimizer(firstMomentDecay, secondMomentDecay, epsilon);
}
