namespace Daylin.NeuralNetworks;

/// <summary>
/// Assertions over the values a neural network produces.
/// </summary>
/// <remarks>
/// Expected values are computed in double precision from the arithmetic a layer is specified to perform, and
/// compared against the network's single-precision results within a tolerance.
/// </remarks>
internal static class NetworkAssert
{
    #region Public Static

    /// <summary>
    /// Default tolerance for a comparison between a computed value and a value the network produced.
    /// </summary>
    /// <remarks>
    /// Single precision carries about seven significant decimal digits, and the values compared here are the
    /// result of a handful of operations on numbers near one.
    /// </remarks>
    public static double Tolerance => 1e-5;

    /// <summary>
    /// Asserts that a network calculates the expected output for an input.
    /// </summary>
    /// <param name="network">
    /// The network to calculate output from.
    /// </param>
    /// <param name="input">
    /// Input values.
    /// </param>
    /// <param name="expectedOutput">
    /// The output values the network should calculate.
    /// </param>
    /// <returns>
    /// Returns a task representing this <c>async</c> operation.
    /// </returns>
    public static async Task OutputIsAsync(
        INeuralNetwork network,
        IEnumerable<float> input,
        params IEnumerable<double> expectedOutput)
    {
        IEnumerable<float> output = await network.CalculateOutputAsync(input);

        ValuesAre(output, expectedOutput, "output");
    }

    /// <summary>
    /// Asserts that a sequence of values holds the expected values, within <see cref="Tolerance"/>.
    /// </summary>
    /// <param name="actualValues">
    /// The values to check.
    /// </param>
    /// <param name="expectedValues">
    /// The values expected.
    /// </param>
    /// <param name="description">
    /// A description of what the values are, used in the message of a failed assertion.
    /// </param>
    public static void ValuesAre(
        IEnumerable<float> actualValues,
        IEnumerable<double> expectedValues,
        string description)
    {
        IReadOnlyList<float> actual = actualValues.ToList();
        IReadOnlyList<double> expected = expectedValues.ToList();

        Assert.AreEqual(expected.Count, actual.Count, $"Count of {description} values.");

        for (int index = 0; index < expected.Count; index++)
        {
            Assert.AreEqual(
                expected[index],
                actual[index],
                Tolerance,
                $"The {description} value at index {index} ({Describe(actual)}, expected {Describe(expected)}).");
        }
    }

    #endregion

    #region Private Static

    private static string Describe<T>(IEnumerable<T> values) => string.Join(", ", values);

    #endregion
}
