using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Activation;

/// <summary>
/// Rectified linear unit (ReLU) function.
/// </summary>
public sealed class ReluActivationFunction : IActivationFunction
{
    public ReluActivationFunction()
    {
    }

    [JsonIgnore]
    public string ShortName => "ReLU";

    [JsonIgnore]
    public string LongName => "Rectified Linear Unit";

    /// <summary>
    /// Rectified linear unit (ReLU) function.
    /// </summary>
    /// <param name="value">
    /// A pre-activated output value for a neuron.
    /// </param>
    /// <returns>
    /// Returns the ReLU and its derivative for <paramref name="value"/>.
    /// </returns>
    public (float Activation, float Derivative) Calculate(float value)
    {
        return (CalculateActivation(value), CalculateDerivative(value));
    }

    /// <summary>
    /// Rectified linear unit (ReLU) function.
    /// </summary>
    /// <param name="value">
    /// A pre-activated output value for a neuron.
    /// </param>
    /// <returns>
    /// Returns the ReLU for <paramref name="value"/>.
    /// </returns>
    public float CalculateActivation(float value)
    {
        return Math.Max(value, 0);
    }

    /// <summary>
    /// Rectified linear unit (ReLU) derivative function.
    /// </summary>
    /// <param name="value">
    /// A pre-activated output value for a neuron.
    /// </param>
    /// <returns>
    /// Returns the derivative of the ReLU for <paramref name="value"/>.
    /// </returns>
    public float CalculateDerivative(float value)
    {
        return value <= 0 ? 0 : 1;
    }
}