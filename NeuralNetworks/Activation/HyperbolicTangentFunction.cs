using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Activation;

/// <summary>
/// Hyperbolic tangent function.
/// </summary>
/// <remarks>
/// The range of the hyperbolic tangent function is (-1, 1).
/// </remarks>
public sealed class HyperbolicTangentFunction : IActivationFunction
{
    public HyperbolicTangentFunction()
    {
    }

    [JsonIgnore]
    public string ShortName => "Tanh";

    [JsonIgnore]
    public string LongName => "Hyperbolic Tangent";

    /// <summary>
    /// Hyperbolic tangent function.
    /// </summary>
    /// <param name="value">
    /// A pre-activated output value for a neuron.
    /// </param>
    /// <returns>
    /// Returns the hyperbolic tangent and its derivative for <paramref name="value"/>.
    /// </returns>
    public (float Activation, float Derivative) Calculate(float value)
    {
        float tanhValue = CalculateActivation(value);

        float derivative = (float)(1 - Math.Pow(tanhValue, 2));

        return (tanhValue, derivative);
    }

    /// <summary>
    /// Hyperbolic tangent function.
    /// </summary>
    /// <param name="value">
    /// A pre-activated output value for a neuron.
    /// </param>
    /// <returns>
    /// Returns the hyperbolic tangent for <paramref name="value"/>, in the range (-1, 1).
    /// </returns>
    public float CalculateActivation(float value)
    {
        return (float)Math.Tanh(value);
    }

    /// <summary>
    /// Hyperbolic tangent derivative function.
    /// </summary>
    /// <param name="value">
    /// A pre-activated output value for a neuron.
    /// </param>
    /// <returns>
    /// Returns the derivative of the hyperbolic tangent for <paramref name="value"/>.
    /// </returns>
    public float CalculateDerivative(float value)
    {
        (float _, float derivative) = Calculate(value);

        return derivative;
    }
}

