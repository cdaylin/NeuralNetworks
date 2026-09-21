using Daylin.Utilities.Exceptions;

namespace Daylin.NeuralNetworks.WeightInitialization;

/// <summary>
/// Tests the weights the initializers generate.
/// </summary>
/// <remarks>
/// An initializer draws weights from a range derived from a neuron's connection counts.  Glorot's range is
/// derived from the fan-in and the fan-out together; He's, from the fan-in alone.  A seed makes the draw
/// reproducible, which is what lets training be repeated exactly.
/// </remarks>
[TestClass]
public class WeightInitializerTests
{
    #region Private Static

    private static int WeightCount => 1000;

    private static IReadOnlyList<float> GenerateWeights(
        IConnectionWeightInitializer initializer,
        int inConnectionCount = 20,
        int outConnectionCount = 20)
    {
        return Enumerable.Range(0, WeightCount)
            .Select(index => initializer.GenerateWeight(inConnectionCount, outConnectionCount))
            .ToList();
    }

    private static IConnectionWeightInitializer CreateInitializer(string algorithmName, int? seed) =>
        algorithmName switch
        {
            "Glorot" => new GlorotUniformWeightInitializer(seed: seed),
            "He" => new HeUniformWeightInitializer(seed: seed),
            _ => throw UnhandledCaseException.Create(algorithmName),
        };

    #endregion

    #region Public

    [TestMethod]
    [DataRow("Glorot")]
    [DataRow("He")]
    public void TwoInitializersGivenOneSeedGenerateTheSameWeights(string algorithmName)
    {
        IReadOnlyList<float> weights = GenerateWeights(CreateInitializer(algorithmName, seed: 7));
        IReadOnlyList<float> sameSeed = GenerateWeights(CreateInitializer(algorithmName, seed: 7));

        Assert.IsTrue(weights.SequenceEqual(sameSeed), "Weights generated from the same seed.");
    }

    [TestMethod]
    [DataRow("Glorot")]
    [DataRow("He")]
    public void TwoInitializersGivenDifferentSeedsGenerateDifferentWeights(string algorithmName)
    {
        IReadOnlyList<float> weights = GenerateWeights(CreateInitializer(algorithmName, seed: 7));
        IReadOnlyList<float> otherSeed = GenerateWeights(CreateInitializer(algorithmName, seed: 8));

        Assert.IsFalse(weights.SequenceEqual(otherSeed), "Weights generated from different seeds.");
    }

    [TestMethod]
    [DataRow("Glorot")]
    [DataRow("He")]
    public void AnInitializerGivenNoSeedDrawsFromAGeneratorItDoesNotOwn(string algorithmName)
    {
        IReadOnlyList<float> weights = GenerateWeights(CreateInitializer(algorithmName, seed: null));
        IReadOnlyList<float> otherWeights = GenerateWeights(CreateInitializer(algorithmName, seed: null));

        Assert.IsFalse(weights.SequenceEqual(otherWeights), "Weights generated without a seed.");
    }

    [TestMethod]
    public void GlorotWeightsLieWithinTheRangeItsFanInAndFanOutGive()
    {
        GlorotUniformWeightInitializer initializer = new(seed: 7);

        float range = initializer.CalculateRange(inConnectionCount: 20, outConnectionCount: 20);

        AssertWeightsFillRange(GenerateWeights(initializer), range);
    }

    [TestMethod]
    public void HeWeightsLieWithinTheRangeItsFanInGives()
    {
        HeUniformWeightInitializer initializer = new(seed: 7);

        float range = initializer.CalculateRange(inConnectionCount: 20);

        AssertWeightsFillRange(GenerateWeights(initializer), range);
    }

    [TestMethod]
    [DataRow("Glorot")]
    [DataRow("He")]
    public void TheMaximumRangeCapsWeightsGeneratedForAFewConnections(string algorithmName)
    {
        float maxRange = 0.1f;

        IConnectionWeightInitializer initializer = algorithmName == "Glorot"
            ? new GlorotUniformWeightInitializer(maxRange, seed: 7)
            : new HeUniformWeightInitializer(maxRange, seed: 7);

        // Two connections would otherwise give a range far wider than the maximum.
        AssertWeightsFillRange(
            GenerateWeights(initializer, inConnectionCount: 2, outConnectionCount: 2),
            maxRange);
    }

    [TestMethod]
    public void TheFanOutMovesGlorotsRangeAndLeavesHesAlone()
    {
        GlorotUniformWeightInitializer glorot = new(seed: 7);

        Assert.AreNotEqual(
            glorot.CalculateRange(inConnectionCount: 20, outConnectionCount: 20),
            glorot.CalculateRange(inConnectionCount: 20, outConnectionCount: 200),
            "Glorot's range over two fan-out values.");

        IReadOnlyList<float> heWeights =
            GenerateWeights(new HeUniformWeightInitializer(seed: 7), inConnectionCount: 20, outConnectionCount: 20);

        IReadOnlyList<float> heWeightsWithWiderFanOut =
            GenerateWeights(new HeUniformWeightInitializer(seed: 7), inConnectionCount: 20, outConnectionCount: 200);

        Assert.IsTrue(
            heWeights.SequenceEqual(heWeightsWithWiderFanOut),
            "He's weights over two fan-out values.");
    }

    #endregion

    #region Private

    /// <summary>
    /// Asserts that weights lie within a range centred on zero, and that they are spread across it.
    /// </summary>
    /// <remarks>
    /// The spread is what separates a correct range from one that has collapsed towards zero, which lying
    /// within the range alone would not catch.
    /// </remarks>
    /// <param name="weights">
    /// The weights to check.
    /// </param>
    /// <param name="range">
    /// The width of the range the weights are drawn from.
    /// </param>
    private void AssertWeightsFillRange(IReadOnlyList<float> weights, float range)
    {
        float limit = range / 2;

        Assert.IsTrue(weights.All(weight => weight >= -limit && weight < limit), $"Weights within ±{limit}.");
        Assert.IsTrue(weights.Min() < -0.4 * limit, "Smallest weight generated.");
        Assert.IsTrue(weights.Max() > 0.4 * limit, "Largest weight generated.");
    }

    #endregion
}
