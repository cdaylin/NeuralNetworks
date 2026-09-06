using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Activation;

/// <summary>
/// An activation function and its derivative function.
/// </summary>
[JsonPolymorphic]
[JsonDerivedType(typeof(EanafFunction), typeDiscriminator: "EANAF")]
[JsonDerivedType(typeof(HyperbolicTangentFunction), typeDiscriminator: "Tanh")]
[JsonDerivedType(typeof(ReluActivationFunction), typeDiscriminator: "ReLU")]
[JsonDerivedType(typeof(SigmoidActivationFunction), typeDiscriminator: "Sigmoid")]
[JsonDerivedType(typeof(SoftsignActivationFunction), typeDiscriminator: "Softsign")]
public interface IActivationFunction
{
    /// <summary>
    /// Abbreviated name for the activation function.  (E.g.:  "ReLU".)
    /// </summary>
    string ShortName { get; }

    /// <summary>
    /// Full name for the activation function.  (E.g.:  "Rectified Linear Unit".)
    /// </summary>
    string LongName { get; }

    /// <summary>
    /// Activation function.
    /// </summary>
    /// <param name="value">
    /// A pre-activated value for a neuron in a neural network.
    /// </param>
    /// <returns>
    /// Returns the result of applying the activation function to <paramref name="value"/>.
    /// </returns>
    float CalculateActivation(float value);

    /// <summary>
    /// Activation function derivative.
    /// </summary>
    /// <param name="value">
    /// A pre-activated value for a neuron in a neural network.
    /// </param>
    /// <returns>
    /// Returns the derivative of the activation function for <paramref name="value"/>.
    /// </returns>
    float CalculateDerivative(float value);

    /// <summary>
    /// Calculates the activation result and its derivative.
    /// </summary>
    /// <remarks>
    /// For some activation functions, it is more efficient to calculate the derivative using the result of the
    /// activation function or one of its intermediate calculations.  When both the activation result and its
    /// derivative are needed, it can sometimes be more efficient to call this method than to call
    /// <see cref="CalculateActivation(float)"/> and <see cref="CalculateDerivative(float)"/> independently.
    /// </remarks>
    /// <param name="value">
    /// A pre-activated value for a neuron in a neural network.
    /// </param>
    /// <returns>
    /// Returns a tuple containing the result of the activation function and its derivative for
    /// <paramref name="value"/>.
    /// </returns>
    (float Activation, float Derivative) Calculate(float value);
}
