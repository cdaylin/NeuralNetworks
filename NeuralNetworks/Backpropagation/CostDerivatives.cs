using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Backpropagation;

/// <summary>
/// Calculates the partial derivative of a cost function with respect to one of a network's output values.
/// </summary>
/// <remarks>
/// A network is trained to minimize a cost function (aka: loss function), which measures the error between the
/// actual output and the desired output for a training sample.  Training moves each parameter <em>against</em>
/// the derivative of that cost with respect to it, and the derivative at the output is where the chain of them
/// starts.
/// <para>
/// This is the derivative itself rather than its negative:  a value larger than its target has a positive
/// derivative, and moving against it makes the value smaller.
/// </para><para>
/// An implementation is a value rather than a function so that a training session can record which cost it was
/// training against, and refuse to resume against another.
/// </para>
/// </remarks>
[JsonPolymorphic]
[JsonDerivedType(typeof(MeanSquaredErrorDerivative), typeDiscriminator: "MeanSquaredError")]
public interface ICostDerivative
{
    /// <summary>
    /// Abbreviated name for the cost function this is the derivative of.
    /// </summary>
    string ShortName { get; }

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
    float Calculate(float targetValue, float actualValue);
}

/// <summary>
/// The derivative of the quadratic cost function (i.e.: mean squared error, MSE).
/// </summary>
/// <remarks>
/// Returns <c>actualValue - targetValue</c>, which is the partial derivative of
/// <c>(targetValue - actualValue)² / 2</c> with respect to the actual value.  The half is what leaves the
/// residual with no coefficient on it; a cost function of <c>(targetValue - actualValue)²</c> would give twice
/// this, which differs only by a constant factor and is therefore absorbed into the learning rate.
/// </remarks>
internal sealed class MeanSquaredErrorDerivative : ICostDerivative
{
    /// <inheritdoc cref="ICostDerivative.ShortName"/>
    public string ShortName => "MeanSquaredError";

    /// <inheritdoc cref="ICostDerivative.Calculate(float, float)"/>
    public float Calculate(float targetValue, float actualValue) => actualValue - targetValue;
}

/// <summary>
/// Commonly used cost derivatives for training a neural network by backpropagation.
/// </summary>
public static class CostDerivatives
{
    /// <summary>
    /// The derivative of the quadratic cost function (i.e.: mean squared error, MSE).
    /// </summary>
    public static ICostDerivative MeanSquaredError { get; } = new MeanSquaredErrorDerivative();
}
