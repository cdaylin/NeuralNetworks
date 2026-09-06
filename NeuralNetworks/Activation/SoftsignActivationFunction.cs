using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Activation;

/// <summary>
/// Softsign function.
/// </summary>
/// <remarks>
/// The range of the softsign function is (-1, 1).
/// </remarks>
public sealed class SoftsignActivationFunction : IActivationFunction
{
    public SoftsignActivationFunction()
    {
    }

    [JsonIgnore]
    public string ShortName => "Softsign";

    [JsonIgnore]
    public string LongName => "Softsign";

    /// <summary>
    /// Softsign function.
    /// </summary>
    /// <param name="value">
    /// A pre-activated output value for a neuron.
    /// </param>
    /// <returns>
    /// Returns the softsign and its derivative for <paramref name="value"/>.
    /// </returns>
    public (float Activation, float Derivative) Calculate(float value)
    {
        return (CalculateActivation(value), CalculateDerivative(value));
    }

    /// <summary>
    /// Softsign function.
    /// </summary>
    /// <param name="value">
    /// A pre-activated output value for a neuron.
    /// </param>
    /// <returns>
    /// Returns the softsign for <paramref name="value"/>, in the range (-1, 1).
    /// </returns>
    public float CalculateActivation(float value)
    {
        return value / (1 + Math.Abs(value));
    }

    /// <summary>
    /// Softsign derivative function.
    /// </summary>
    /// <param name="value">
    /// A pre-activated output value for a neuron.
    /// </param>
    /// <returns>
    /// Returns the derivative of the softsign for <paramref name="value"/>.
    /// </returns>
    public float CalculateDerivative(float value)
    {
        return (float)(1 / Math.Pow(Math.Abs(value) + 1, 2));
    }
}
