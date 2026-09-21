namespace Daylin.NeuralNetworks.Backpropagation.Optimization;

/// <summary>
/// An optimizer that accumulates a velocity for each parameter, so that gradients agreeing from step to step
/// build on one another and gradients that disagree cancel.
/// </summary>
/// <remarks>
/// Gradient descent with momentum:  <c>v ← μ·v + g</c>, then <c>θ ← θ - α·v</c>, for a parameter <c>θ</c>, a
/// learning rate <c>α</c>, a gradient <c>g</c> and a momentum <c>μ</c>.  Each velocity starts at zero.
/// </remarks>
public sealed class MomentumOptimizer : IOptimizer
{
    #region Construction

    /// <summary>
    /// Constructs an optimizer.
    /// </summary>
    /// <param name="momentum">
    /// The fraction of a parameter's velocity that survives a step.  (See <see cref="Momentum"/>.)
    /// </param>
    public MomentumOptimizer(float momentum = 0.9f)
    {
        momentum.ThrowIfNegative();
        momentum.ThrowIfGreaterThanOrEqualTo(1);

        Momentum = momentum;
    }

    #endregion

    #region Public

    /// <inheritdoc cref="IOptimizer.ShortName"/>
    public string ShortName => "Momentum";

    /// <inheritdoc cref="IOptimizer.LongName"/>
    public string LongName => "Gradient Descent with Momentum";

    /// <summary>
    /// The fraction of a parameter's velocity that survives a step, in the range [0, 1).
    /// </summary>
    /// <remarks>
    /// A momentum of zero leaves the optimizer equivalent to plain gradient descent.
    /// </remarks>
    public float Momentum { get; }

    /// <inheritdoc cref="IOptimizer.CreateState(int)"/>
    public IOptimizerState? CreateState(int parameterCount) => new MomentumOptimizerState(parameterCount);

    /// <inheritdoc cref="IOptimizer.CanApplyTo(IOptimizerState?, int)"/>
    public bool CanApplyTo(IOptimizerState? state, int parameterCount) =>
        state is MomentumOptimizerState && state.ParameterCount == parameterCount;

    /// <inheritdoc cref="IOptimizer.ApplyGradients(IOptimizerState?, float[], IReadOnlyList{float}, float)"/>
    public void ApplyGradients(
        IOptimizerState? state,
        float[] parameters,
        IReadOnlyList<float> gradients,
        float learningRate)
    {
        parameters.ThrowIfNull();
        gradients.Count.ThrowIfNotEqualTo(parameters.Length);

        float[] velocities = ((MomentumOptimizerState)state!).Velocities;

        for (int index = 0; index < parameters.Length; index++)
        {
            velocities[index] = (Momentum * velocities[index]) + gradients[index];

            parameters[index] -= learningRate * velocities[index];
        }
    }

    #endregion
}
