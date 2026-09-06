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
    public (float Activation, float Derivative) Calculate(float value)
    {
        return (CalculateActivation(value), CalculateDerivative(value));
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

        double power = Math.Exp(value);

        return (float)(value * power / (power + 2));
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

        double power = Math.Exp(value);

        return (float)((Math.Pow(power, 2) + power * 2 * (value + 1)) / Math.Pow(power + 2, 2));
    }

    private static float MaxThreshold => 18f;
}
