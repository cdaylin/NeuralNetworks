namespace Daylin.NeuralNetworks.Backpropagation.Cost;

/// <summary>
/// The quadratic cost function (i.e.:  mean squared error, MSE).
/// </summary>
/// <remarks>
/// The cost of one output value is <c>(targetValue - actualValue)² / 2</c>, and its derivative with respect to
/// the actual value is therefore <c>actualValue - targetValue</c>.  The half is what leaves the residual with no
/// coefficient on it;  a cost function of <c>(targetValue - actualValue)²</c> would give twice this, which
/// differs only by a constant factor and is therefore absorbed into the learning rate.
/// </remarks>
public sealed class MeanSquaredErrorCostFunction : ICostFunction
{
    #region Public

    /// <inheritdoc cref="ICostFunction.ShortName"/>
    public string ShortName => Discriminator;

    /// <inheritdoc cref="ICostFunction.Calculate(float, float)"/>
    public float Calculate(float targetValue, float actualValue)
    {
        float residual = targetValue - actualValue;

        return residual * residual / 2;
    }

    /// <inheritdoc cref="ICostFunction.CalculateDerivative(float, float)"/>
    public float CalculateDerivative(float targetValue, float actualValue) => actualValue - targetValue;

    #endregion

    #region Internal

    /// <summary>
    /// The name this cost function is written under in a saved session, and its <see cref="ShortName"/>.
    /// </summary>
    internal const string Discriminator = "MeanSquaredError";

    #endregion
}
