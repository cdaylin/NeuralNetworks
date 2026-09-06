namespace Daylin.NeuralNetworks.Backpropagation;

/// <summary>
/// Calculates the direction and magnitude by which an output value should move to reduce a cost function
/// measuring the error between the actual output and the desired output for a training sample.
/// </summary>
/// <remarks>
/// This is the <em>negative</em> partial derivative of the cost function with respect to the actual value --
/// the direction of steepest descent rather than of steepest ascent -- so that training moves along it
/// directly instead of against it.  The distinction matters when implementing one:  a function returning
/// the partial derivative itself trains the network away from its target.
/// </remarks>
/// <param name="targetValue">
/// Desired output value.
/// </param>
/// <param name="actualValue">
/// Actual output value.
/// </param>
/// <returns>
/// Returns the negative partial derivative of the cost function with respect to <paramref name="actualValue"/>.
/// </returns>
public delegate float ErrorFunction(float targetValue, float actualValue);

/// <summary>
/// Commonly used functions for calculating output errors for training a neural network using backpropagation.
/// </summary>
/// <remarks>
/// A neural network is trained to minimize a cost function (aka: loss function), which measures the error
/// between the actual output and the desired output for a training sample.  An error function is the derivative
/// of a cost function.
/// <para>
/// For example, to train a neural network to minimize the mean squared error (MSE) (aka: quadratic cost function)
/// the error function used to train the neural network should be the derivative of the mean squared error.
/// </para>
/// </remarks>
public static class ErrorFunctions
{
    /// <summary>
    /// The error function for the quadratic cost function (i.e.: mean squared error, MSE).
    /// </summary>
    /// <remarks>
    /// Returns <c>targetValue - actualValue</c>, which is the negative partial derivative of
    /// <c>(targetValue - actualValue)² / 2</c> with respect to <paramref name="actualValue"/>.  The half
    /// is what leaves the residual with no coefficient on it; a cost function of
    /// <c>(targetValue - actualValue)²</c> would give <c>2 * (targetValue - actualValue)</c>, which differs
    /// only by a constant factor and is therefore absorbed into the learning rate.
    /// </remarks>
    /// <param name="targetValue">
    /// The target (i.e.: desired, expected) result.
    /// </param>
    /// <param name="actualValue">
    /// The actual result.
    /// </param>
    public static float MeanSquaredError(float targetValue, float actualValue) => targetValue - actualValue;
}