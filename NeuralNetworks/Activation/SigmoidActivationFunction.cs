using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Activation;

/// <summary>
/// Logistic sigmoid function.
/// </summary>
/// <remarks>
/// The range of the logistic sigmoid function is (0, 1).
/// </remarks>
public sealed class SigmoidActivationFunction : IActivationFunction
{
    public SigmoidActivationFunction()
    {
    }

    [JsonIgnore]
    public string ShortName => "Sigmoid";

    [JsonIgnore]
    public string LongName => "Logistic Sigmoid";

    /// <summary>
    /// Logistic sigmoid function.
    /// </summary>
    /// <param name="value">
    /// A pre-activated output value for a neuron.
    /// </param>
    /// <returns>
    /// Returns the logistic sigmoid and its derivative for <paramref name="value"/>.
    /// </returns>
    public (float Activation, float Derivative) Calculate(float value)
    {
        float sigmoidValue = CalculateActivation(value);

        float derivative = sigmoidValue * (1 - sigmoidValue);

        return (sigmoidValue, derivative);
    }

    /// <summary>
    /// Logistic sigmoid function.
    /// </summary>
    /// <param name="value">
    /// A pre-activated output value for a neuron.
    /// </param>
    /// <returns>
    /// Returns the logistic sigmoid for <paramref name="value"/>, in the range (0, 1).
    /// </returns>
    public float CalculateActivation(float value)
    {
        return (float)(1 / (1 + Math.Exp(-value)));
    }

    /// <summary>
    /// Logistic sigmoid derivative function.
    /// </summary>
    /// <param name="value">
    /// A pre-activated output value for a neuron.
    /// </param>
    /// <returns>
    /// Returns the derivative of the logistic sigmoid for <paramref name="value"/>.
    /// </returns>
    public float CalculateDerivative(float value)
    {
        (float _, float derivative) = Calculate(value);

        return derivative;
    }
}
