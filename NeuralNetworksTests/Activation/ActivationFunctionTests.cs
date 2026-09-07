using Daylin.Utilities.Exceptions;

namespace Daylin.NeuralNetworks.Activation;

/// <summary>
/// Tests for the activation functions.
/// </summary>
/// <remarks>
/// <see cref="IActivationFunction"/> offers three ways to ask for the same information:
/// <see cref="IActivationFunction.CalculateActivation(float)"/>,
/// <see cref="IActivationFunction.CalculateDerivative(float)"/>, and
/// <see cref="IActivationFunction.Calculate(float)"/> for both at once.  The combined method exists so an
/// implementation can share work between the two -- reusing the activation result, or an intermediate
/// calculation such as an exponential -- which means it is free to compute its results by a different route
/// than the single-value methods do.
/// <para>
/// That freedom is the hazard these tests exist for:  an implementation that shares work incorrectly returns
/// answers that differ depending on which method was called, and nothing else in the library would notice.
/// Training uses the combined method exclusively, so a wrong answer there would be silently trained on while
/// every direct call kept returning the right one.
/// </para>
/// </remarks>
[TestClass]
public class ActivationFunctionTests
{
    /// <summary>
    /// Verifies that the combined method returns exactly what the single-value methods return.
    /// </summary>
    /// <remarks>
    /// Exact equality rather than a tolerance, deliberately.  These are three routes to one value, not three
    /// approximations of it, so any difference at all is a defect rather than accumulated error.
    /// </remarks>
    /// <param name="activationFunctionName">
    /// Abbreviated name of the activation function under test.
    /// </param>
    [TestMethod]
    [DataRow("EANAF")]
    [DataRow("ReLU")]
    [DataRow("Tanh")]
    [DataRow("Softsign")]
    [DataRow("Sigmoid")]
    public void TestCombinedResultMatchesSingleValueResults(string activationFunctionName)
    {
        IActivationFunction activationFunction = GetActivationFunction(activationFunctionName);

        foreach (float input in EnumerateInputs())
        {
            (float activation, float derivative) = activationFunction.Calculate(input);

            Assert.IsTrue(
                activationFunction.CalculateActivation(input).Equals(activation),
                $"The activation results disagreed for an input of {input}.");

            Assert.IsTrue(
                activationFunction.CalculateDerivative(input).Equals(derivative),
                $"The derivative results disagreed for an input of {input}.");
        }
    }

    /// <summary>
    /// Enumerates the inputs every activation function is tested against.
    /// </summary>
    /// <remarks>
    /// Dense across the range where the functions curve, and around zero, plus values chosen to sit either
    /// side of the threshold at which EANAF switches to its linear approximation -- a branch that a shared
    /// intermediate calculation could easily be taken past.
    /// </remarks>
    /// <returns>
    /// Returns the inputs to test.
    /// </returns>
    private static IEnumerable<float> EnumerateInputs()
    {
        for (float input = -25f; input <= 25f; input += 0.001f)
            yield return input;

        yield return 0f;
        yield return 17.999f;
        yield return 18f;
        yield return 18.001f;
        yield return float.Epsilon;
        yield return -float.Epsilon;
        yield return 1000f;
        yield return -1000f;
    }

    private static IActivationFunction GetActivationFunction(string shortName)
    {
        return shortName switch
        {
            "EANAF" => ActivationFunctions.Eanaf,
            "ReLU" => ActivationFunctions.Relu,
            "Tanh" => ActivationFunctions.Tanh,
            "Softsign" => ActivationFunctions.Softsign,
            "Sigmoid" => ActivationFunctions.Sigmoid,
            _ => throw UnhandledCaseException.Create(shortName),
        };
    }
}
