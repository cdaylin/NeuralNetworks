using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Backpropagation.Cost;

/// <summary>
/// Measures how far one of a network's output values is from the value wanted, and how the cost changes with it.
/// </summary>
/// <remarks>
/// A network is trained to minimize a cost function (aka:  loss function), which measures the error between the
/// actual output and the desired output for a training sample.  Training moves each parameter <em>against</em>
/// the derivative of that cost with respect to it, and the derivative at the output is where the chain of them
/// starts;  the cost itself is what a caller watches to see whether training is working.
/// <para>
/// <see cref="CalculateDerivative"/> returns the derivative itself rather than its negative:  a value larger
/// than its target has a positive derivative, and moving against it makes the value smaller.
/// </para><para>
/// An implementation is a value rather than a function so that a training session can record which cost it was
/// training against, and refuse to resume against another.
/// </para>
/// </remarks>
[JsonPolymorphic]
[JsonDerivedType(typeof(MeanSquaredErrorCostFunction), typeDiscriminator: MeanSquaredErrorCostFunction.Discriminator)]
public interface ICostFunction
{
    /// <summary>
    /// Abbreviated name for the cost function.
    /// </summary>
    string ShortName { get; }

    /// <summary>
    /// Calculates the cost of one output value.
    /// </summary>
    /// <param name="targetValue">
    /// Desired output value.
    /// </param>
    /// <param name="actualValue">
    /// Actual output value.
    /// </param>
    /// <returns>
    /// Returns the cost, which is zero where the values are equal and grows as they part.
    /// </returns>
    float Calculate(float targetValue, float actualValue);

    /// <summary>
    /// Calculates the partial derivative of the cost with respect to an output value.
    /// </summary>
    /// <param name="targetValue">
    /// Desired output value.
    /// </param>
    /// <param name="actualValue">
    /// Actual output value.
    /// </param>
    /// <returns>
    /// Returns the partial derivative of the cost with respect to <paramref name="actualValue"/>.
    /// </returns>
    float CalculateDerivative(float targetValue, float actualValue);
}
