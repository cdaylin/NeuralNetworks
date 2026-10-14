namespace Daylin.NeuralNetworks.Backpropagation.Optimization;

/// <summary>
/// An optimizer that scales each parameter's step by how consistent and how large its recent gradients have
/// been.
/// </summary>
/// <remarks>
/// Adam (adaptive moment estimation) keeps two decaying means for each parameter:  of its gradients, and of
/// their squares.  A parameter moves against the first divided by the square root of the second, so that a
/// parameter whose gradients agree takes a step near the full learning rate, while one whose gradients are
/// large but inconsistent takes a smaller one.
/// <para>
/// For a gradient <c>g</c>, decay rates <c>β₁</c> and <c>β₂</c>, an epsilon <c>ε</c> and a step count
/// <c>t</c>:  <c>m ← β₁·m + (1 - β₁)·g</c> and <c>v ← β₂·v + (1 - β₂)·g²</c>, then
/// <c>θ ← θ - α·m̂ / (√v̂ + ε)</c>, where <c>m̂ = m / (1 - β₁ᵗ)</c> and <c>v̂ = v / (1 - β₂ᵗ)</c>.
/// </para><para>
/// Both means start at zero, which biases them towards it while few steps have been taken.  Each is therefore
/// divided by the remainder of its decay over the steps so far before it is used, which leaves the first step
/// close to the learning rate however the decay rates are set.
/// </para>
/// </remarks>
public sealed class AdamOptimizer : IOptimizer
{
    #region Construction

    /// <summary>
    /// Constructs an optimizer.
    /// </summary>
    /// <param name="firstMomentDecay">
    /// The fraction of the mean gradient that survives a step.  (See <see cref="FirstMomentDecay"/>.)
    /// </param>
    /// <param name="secondMomentDecay">
    /// The fraction of the mean squared gradient that survives a step.  (See
    /// <see cref="SecondMomentDecay"/>.)
    /// </param>
    /// <param name="epsilon">
    /// A small value added to the denominator of a step.  (See <see cref="Epsilon"/>.)
    /// </param>
    public AdamOptimizer(float firstMomentDecay = 0.9f, float secondMomentDecay = 0.999f, float epsilon = 1e-8f)
    {
        firstMomentDecay.ThrowIfNegative();
        firstMomentDecay.ThrowIfGreaterThanOrEqualTo(1);
        secondMomentDecay.ThrowIfNegative();
        secondMomentDecay.ThrowIfGreaterThanOrEqualTo(1);
        epsilon.ThrowIfNotPositive();

        FirstMomentDecay = firstMomentDecay;
        SecondMomentDecay = secondMomentDecay;
        Epsilon = epsilon;
    }

    #endregion

    #region Public

    /// <inheritdoc cref="IOptimizer.ShortName"/>
    public string ShortName => Discriminator;

    /// <inheritdoc cref="IOptimizer.LongName"/>
    public string LongName => "Adaptive Moment Estimation";

    /// <summary>
    /// The fraction of the mean gradient that survives a step, in the range [0, 1).
    /// </summary>
    public float FirstMomentDecay { get; }

    /// <summary>
    /// The fraction of the mean squared gradient that survives a step, in the range [0, 1).
    /// </summary>
    public float SecondMomentDecay { get; }

    /// <summary>
    /// A small positive value added to the denominator of a step, so that a parameter whose gradients have all
    /// been zero takes no step rather than an undefined one.
    /// </summary>
    public float Epsilon { get; }

    /// <inheritdoc cref="IOptimizer.CreateState(int)"/>
    public IOptimizerState? CreateState(int parameterCount) => new AdamOptimizerState(parameterCount);

    /// <inheritdoc cref="IOptimizer.CanApplyTo(IOptimizerState?, int)"/>
    public bool CanApplyTo(IOptimizerState? state, int parameterCount) =>
        OptimizerStateGuard.IsStateFor<AdamOptimizerState>(state, parameterCount);

    /// <inheritdoc cref="IOptimizer.ApplyGradients(IOptimizerState?, float[], ReadOnlySpan{float}, float)"/>
    public void ApplyGradients(
        IOptimizerState? state,
        float[] parameters,
        ReadOnlySpan<float> gradients,
        float learningRate)
    {
        OptimizerStateGuard.RequireOneGradientPerParameter(parameters, gradients);

        AdamOptimizerState adamState =
            OptimizerStateGuard.Require<AdamOptimizerState>(state, parameters.Length, ShortName);

        adamState.StepCount += 1;

        double firstMomentRemainder = 1 - Math.Pow(FirstMomentDecay, adamState.StepCount);
        double secondMomentRemainder = 1 - Math.Pow(SecondMomentDecay, adamState.StepCount);

        for (int index = 0; index < parameters.Length; index++)
        {
            parameters[index] -= CalculateStep(
                adamState,
                index,
                gradients[index],
                firstMomentRemainder,
                secondMomentRemainder,
                learningRate);
        }
    }

    #endregion

    #region Internal

    /// <summary>
    /// The name this optimizer is written under in a saved session, and its <see cref="ShortName"/>.
    /// </summary>
    internal const string Discriminator = "Adam";

    #endregion

    #region Private

    /// <summary>
    /// Advances one parameter's moment estimates and calculates the distance it should move.
    /// </summary>
    /// <param name="state">
    /// The state held for the parameter's group.
    /// </param>
    /// <param name="index">
    /// Index of the parameter within its group.
    /// </param>
    /// <param name="gradient">
    /// The gradient of the cost with respect to the parameter.
    /// </param>
    /// <param name="firstMomentRemainder">
    /// The share of the first moment that the steps so far have not decayed away.
    /// </param>
    /// <param name="secondMomentRemainder">
    /// The share of the second moment that the steps so far have not decayed away.
    /// </param>
    /// <param name="learningRate">
    /// The learning rate for the training epoch in progress.
    /// </param>
    /// <returns>
    /// Returns the distance the parameter should move against its gradient.
    /// </returns>
    private float CalculateStep(
        AdamOptimizerState state,
        int index,
        float gradient,
        double firstMomentRemainder,
        double secondMomentRemainder,
        float learningRate)
    {
        state.FirstMoments[index] =
            (FirstMomentDecay * state.FirstMoments[index]) + ((1 - FirstMomentDecay) * gradient);

        state.SecondMoments[index] =
            (SecondMomentDecay * state.SecondMoments[index]) + ((1 - SecondMomentDecay) * gradient * gradient);

        double correctedFirstMoment = state.FirstMoments[index] / firstMomentRemainder;
        double correctedSecondMoment = state.SecondMoments[index] / secondMomentRemainder;

        return (float)(learningRate * correctedFirstMoment / (Math.Sqrt(correctedSecondMoment) + Epsilon));
    }

    #endregion
}
