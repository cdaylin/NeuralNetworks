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
    public string ShortName => "GradientDescent";

    /// <inheritdoc cref="IOptimizer.LongName"/>
    public string LongName => "Gradient Descent";

    /// <inheritdoc cref="IOptimizer.CreateState(int)"/>
    public IOptimizerState? CreateState(int parameterCount) => null;

    /// <inheritdoc cref="IOptimizer.CanApplyTo(IOptimizerState?, int)"/>
    public bool CanApplyTo(IOptimizerState? state, int parameterCount) => state is null;

    /// <inheritdoc cref="IOptimizer.ApplyGradients(IOptimizerState?, float[], IReadOnlyList{float}, float)"/>
    public void ApplyGradients(
        IOptimizerState? state,
        float[] parameters,
        IReadOnlyList<float> gradients,
        float learningRate)
    {
        parameters.ThrowIfNull();
        gradients.Count.ThrowIfNotEqualTo(parameters.Length);

        for (int index = 0; index < parameters.Length; index++)
            parameters[index] -= learningRate * gradients[index];
    }

    #endregion
}
