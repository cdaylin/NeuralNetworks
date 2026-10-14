namespace Daylin.NeuralNetworks.Backpropagation.Optimization;

/// <summary>
/// An optimizer that moves each parameter against its gradient, scaled by the learning rate.
/// </summary>
/// <remarks>
/// Plain gradient descent:  <c>θ ← θ - α·g</c>, for a parameter <c>θ</c>, a learning rate <c>α</c> and a
/// gradient <c>g</c>.  It keeps no state, so a step depends on the gradients of the batch that produced it and
/// on nothing before it.
/// </remarks>
public sealed class GradientDescentOptimizer : IOptimizer
{
    #region Public

    /// <inheritdoc cref="IOptimizer.ShortName"/>
    public string ShortName => Discriminator;

    /// <inheritdoc cref="IOptimizer.LongName"/>
    public string LongName => "Gradient Descent";

    /// <inheritdoc cref="IOptimizer.CreateState(int)"/>
    public IOptimizerState? CreateState(int parameterCount) => null;

    /// <inheritdoc cref="IOptimizer.CanApplyTo(IOptimizerState?, int)"/>
    public bool CanApplyTo(IOptimizerState? state, int parameterCount) => state is null;

    /// <inheritdoc cref="IOptimizer.ApplyGradients(IOptimizerState?, float[], ReadOnlySpan{float}, float)"/>
    public void ApplyGradients(
        IOptimizerState? state,
        float[] parameters,
        ReadOnlySpan<float> gradients,
        float learningRate)
    {
        OptimizerStateGuard.RequireOneGradientPerParameter(parameters, gradients);

        // Refused rather than ignored, so that state from another optimizer is as loud here as it is in the
        //  optimizers that carry state:  a caller holding such state has made a mistake either way.
        OptimizerStateGuard.RequireNone(state, ShortName);

        for (int index = 0; index < parameters.Length; index++)
            parameters[index] -= learningRate * gradients[index];
    }

    #endregion

    #region Internal

    /// <summary>
    /// The name this optimizer is written under in a saved session, and its <see cref="ShortName"/>.
    /// </summary>
    internal const string Discriminator = "GradientDescent";

    #endregion
}
