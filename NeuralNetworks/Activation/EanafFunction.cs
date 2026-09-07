using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Activation;

/// <summary>
/// Efficient asymmetric nonlinear activation function (EANAF).
/// </summary>
public sealed class EanafFunction : IActivationFunction
{
    public EanafFunction()
    {
    }

    [JsonIgnore]
    public string ShortName => "EANAF";

    [JsonIgnore]
    public string LongName => "Efficient Asymmetric Nonlinear Activation Function";

    /// <summary>
    /// EANAF function.
    /// </summary>
    /// <param name="value">
    /// A pre-activated output value for a neuron.
    /// </param>
    /// <returns>
    /// Returns the activation result and its derivative for <paramref name="value"/>.
    /// </returns>
    /// <remarks>
    /// Both results are derived from a single exponential.  Computing them through
    /// <see cref="CalculateActivation(float)"/> and <see cref="CalculateDerivative(float)"/> instead would
    /// evaluate <see cref="Math.Exp(double)"/> twice for the same input, which is the cost this method
    /// exists to avoid.
    /// </remarks>
    public (float Activation, float Derivative) Calculate(float value)
    {
        if (value > MaxThreshold)
            return (value, 1f);

        double power = Math.Exp(value);

        return (CalculateActivation(value, power), CalculateDerivative(value, power));
    }

    /// <summary>
    /// EANAF function.
    /// </summary>
    /// <param name="value">
    /// A pre-activated output value for a neuron.
    /// </param>
    /// <returns>
    /// Returns the activation result for <paramref name="value"/>.
    /// </returns>
    public float CalculateActivation(float value)
    {
        if (value > MaxThreshold)
            return value;

        return CalculateActivation(value, Math.Exp(value));
    }

    /// <summary>
    /// EANAF derivative function.
    /// </summary>
    /// <param name="value">
    /// A pre-activated output value for a neuron.
    /// </param>
    /// <returns>
    /// Returns the derivative of the activation value for <paramref name="value"/>.
    /// </returns>
    public float CalculateDerivative(float value)
    {
        if (value > MaxThreshold)
            return 1f;

        return CalculateDerivative(value, Math.Exp(value));
    }

    /// <summary>
    /// EANAF function, for a caller that has already evaluated the exponential.
    /// </summary>
    /// <param name="value">
    /// A pre-activated output value for a neuron, at or below <see cref="MaxThreshold"/>.
    /// </param>
    /// <param name="power">
    /// <see cref="Math.Exp(double)"/> of <paramref name="value"/>.
    /// </param>
    /// <returns>
    /// Returns the activation result for <paramref name="value"/>.
    /// </returns>
    private static float CalculateActivation(float value, double power)
    {
        return (float)(value * power / (power + 2));
    }

    /// <summary>
    /// EANAF derivative function, for a caller that has already evaluated the exponential.
    /// </summary>
    /// <param name="value">
    /// A pre-activated output value for a neuron, at or below <see cref="MaxThreshold"/>.
    /// </param>
    /// <param name="power">
    /// <see cref="Math.Exp(double)"/> of <paramref name="value"/>.
    /// </param>
    /// <returns>
    /// Returns the derivative of the activation value for <paramref name="value"/>.
    /// </returns>
    private static float CalculateDerivative(float value, double power)
    {
        double denominator = power + 2;

        return (float)(((power * power) + (power * 2 * (value + 1))) / (denominator * denominator));
    }

    private static float MaxThreshold => 18f;
}
