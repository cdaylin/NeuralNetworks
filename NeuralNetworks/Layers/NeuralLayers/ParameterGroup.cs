using Daylin.NeuralNetworks.Backpropagation.Optimization;

namespace Daylin.NeuralNetworks.Layers.NeuralLayers;

/// <summary>
/// Parameters a layer holds together, with the cost gradients accumulated for them and the state its optimizer
/// carries.
/// </summary>
/// <remarks>
/// A group is a layer's bias vector, or the connection weights of one of its output values.  Grouping them is
/// what lets an optimizer that keeps state per parameter -- a velocity, a moment estimate -- hold that state
/// in arrays the same shape as the parameters themselves.
/// <para>
/// A group holds the parameters by reference, so moving them moves the layer's own weights and biases.  A
/// convolutional layer's shared weights are one group however many output values use them, which is what makes
/// every span's gradient accumulate into the same place.
/// </para>
/// </remarks>
internal sealed class ParameterGroup
{
    #region Construction

    public ParameterGroup(float[] parameters)
    {
        parameters.ThrowIfNull();
        parameters.Length.ThrowIfNotPositive();

        Parameters = parameters;
        Gradients = new float[parameters.Length];
    }

    #endregion

    #region Public

    /// <summary>
    /// The parameters themselves, held by reference.
    /// </summary>
    public float[] Parameters { get; }

    /// <summary>
    /// The gradient accumulated for each parameter since the last time they moved.
    /// </summary>
    public float[] Gradients { get; }

    /// <summary>
    /// State the optimizer carries for this group, or <see langword="null"/> where it carries none.
    /// </summary>
    public IOptimizerState? OptimizerState { get; set; }

    /// <summary>
    /// Moves the parameters against the gradients accumulated for them, and discards those gradients.
    /// </summary>
    /// <remarks>
    /// The accumulated gradients are divided by the number of samples they came from, so that a batch's step
    /// does not depend on how many samples the batch held.
    /// </remarks>
    /// <param name="optimizer">
    /// The optimizer moving the parameters.
    /// </param>
    /// <param name="learningRate">
    /// The learning rate for the training epoch in progress.
    /// </param>
    /// <param name="sampleCount">
    /// Number of samples the gradients were accumulated from.
    /// </param>
    /// <param name="gradientScale">
    /// The factor every mean gradient is multiplied by before the optimizer sees it:  one, unless the
    /// network's gradients are being scaled down to a maximum norm.
    /// </param>
    public void ApplyGradients(IOptimizer optimizer, float learningRate, int sampleCount, float gradientScale)
    {
        optimizer.ThrowIfNull();
        sampleCount.ThrowIfNotPositive();
        gradientScale.ThrowIfNotPositive();

        if (!optimizer.CanApplyTo(OptimizerState, Parameters.Length))
            OptimizerState = optimizer.CreateState(Parameters.Length);

        optimizer.ApplyGradients(
            OptimizerState,
            Parameters,
            CalculateMeanGradients(sampleCount, gradientScale),
            learningRate);

        DiscardGradients();
    }

    /// <summary>
    /// Calculates this group's share of the norm of the network's gradients.
    /// </summary>
    /// <param name="sampleCount">
    /// Number of samples the gradients were accumulated from.
    /// </param>
    /// <returns>
    /// Returns the sum of the squares of the group's mean gradients.
    /// </returns>
    public double CalculateSquaredGradientNorm(int sampleCount)
    {
        sampleCount.ThrowIfNotPositive();

        return CalculateMeanGradients(sampleCount, gradientScale: 1)
            .Sum(gradient => (double)gradient * gradient);
    }

    /// <summary>
    /// Discards the gradients accumulated for the parameters, leaving the parameters themselves alone.
    /// </summary>
    public void DiscardGradients() => Array.Clear(Gradients);

    #endregion

    #region Private

    private IReadOnlyList<float> CalculateMeanGradients(int sampleCount, float gradientScale)
    {
        if (sampleCount == 1 && gradientScale == 1)
            return Gradients;

        return Gradients.Select(gradient => gradient / sampleCount * gradientScale).ToArray();
    }

    #endregion
}
