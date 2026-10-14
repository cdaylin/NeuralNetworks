using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.Backpropagation.Cost;
using Daylin.NeuralNetworks.Backpropagation.LearningRate;
using Daylin.NeuralNetworks.Backpropagation.Optimization;
using Daylin.NeuralNetworks.Serialization;
using Daylin.NeuralNetworks.WeightInitialization;

namespace Daylin.NeuralNetworks.Backpropagation;

/// <summary>
/// Tests that training is reproducible:  the same starting weights, settings and samples leave the same weights,
/// to the last bit, however many times the training is run.
/// </summary>
/// <remarks>
/// Reproducibility is promised for one build of the library.  It is not promised across releases, nor against
/// the arithmetic of an earlier implementation:  the order in which a sum is taken, or in which two factors are
/// multiplied, moves the last bit, and a later release is free to change either.
/// <para>
/// Each case trains a network holding every layer type, and a convolutional layer whose shared weights are
/// owned by several spans, since those are where a sum has an order to get wrong.  The training runs through
/// epochs, a ragged last batch, and a single sample, under a decaying rate.  Running it several times within
/// one process catches an order that depends on how threads were scheduled; an order that depends on the
/// process is not caught here.
/// </para>
/// </remarks>
[TestClass]
public class ReproducibilityTests
{
    #region Private Static

    private static int RepeatCount => 5;

    #endregion

    #region Public

    [TestMethod]
    [DataRow("GradientDescent", null)]
    [DataRow("Momentum", null)]
    [DataRow("Adam", null)]
    [DataRow("GradientDescent", 0.05f)]
    [DataRow("Adam", 0.05f)]
    public async Task TrainingTheSameNetworkOnTheSameSamplesLeavesTheSameBitsAsync(
        string optimizerName,
        float? maxGradientNorm)
    {
        // A limit on the norm is the one setting that makes every parameter's step depend on every other
        //  parameter's gradient, through a sum over the whole network:  a summation order that varied between
        //  runs would show here and nowhere else.
        IReadOnlyList<int> expectedBits = await TrainAsync(optimizerName, maxGradientNorm);

        for (int repeatIndex = 1; repeatIndex < RepeatCount; repeatIndex++)
        {
            IReadOnlyList<int> actualBits = await TrainAsync(optimizerName, maxGradientNorm);

            Assert.AreEqual(expectedBits.Count, actualBits.Count, "Count of parameters.");

            for (int index = 0; index < expectedBits.Count; index++)
            {
                Assert.AreEqual(
                    expectedBits[index],
                    actualBits[index],
                    $"Bits of parameter {index} on repeat {repeatIndex}.");
            }
        }
    }

    #endregion

    #region Private

    private IOptimizer GetOptimizer(string name)
    {
        return name switch
        {
            "GradientDescent" => Optimizers.GradientDescent,
            "Momentum" => Optimizers.Momentum(),
            "Adam" => Optimizers.Adam(),
            _ => throw new ArgumentOutOfRangeException(nameof(name), name, "Unknown optimizer."),
        };
    }

    /// <summary>
    /// Trains freshly built networks from their seeded starting weights, and reads back every weight and bias.
    /// </summary>
    /// <param name="optimizerName">
    /// The optimizer to train with.
    /// </param>
    /// <param name="maxGradientNorm">
    /// The largest the norm of a batch's gradients may be, or <see langword="null"/> for no limit.
    /// </param>
    /// <returns>
    /// Returns the bit pattern of every weight and bias of every network, in a fixed order.
    /// </returns>
    private async Task<IReadOnlyList<int>> TrainAsync(string optimizerName, float? maxGradientNorm)
    {
        List<int> bits = [];

        foreach (INeuralNetwork network in BuildNetworks())
        {
            TrainingSession session = TrainingSession.Adopt(
                network,
                new TrainingSettings(
                    CostFunctions.MeanSquaredError,
                    GetOptimizer(optimizerName),
                    LearningRateSchedulers.ExponentialDecayLearningRateScheduler(0.05f, decayRate: 0.9f),
                    maxGradientNorm));

            // Seeded, so every repeat draws the same samples.
            Random random = new(7);

            IReadOnlyList<IReadOnlyList<float>> input = CreateSamples(random, network.InputCount);
            IReadOnlyList<IReadOnlyList<float>> targetOutput = CreateSamples(random, network.OutputCount);

            for (int epochIndex = 0; epochIndex < 3; epochIndex++)
                await session.TrainEpochAsync(input, targetOutput, batchSize: 7);

            await session.TrainSampleAsync(input[0], targetOutput[0]);

            foreach (NeuralLayerDocument layer in NetworkDocument.FromNetwork(network).NeuralLayers)
            {
                bits.AddRange(layer.WeightMatrix.SelectMany(row => row).Select(BitConverter.SingleToInt32Bits));
                bits.AddRange(layer.BiasVector.Select(BitConverter.SingleToInt32Bits));
            }
        }

        return bits;
    }

    /// <summary>
    /// Creates twenty samples of values between minus one and one.
    /// </summary>
    /// <param name="random">
    /// The source of the values.
    /// </param>
    /// <param name="valueCount">
    /// Number of values in each sample.
    /// </param>
    /// <returns>
    /// Returns the samples.
    /// </returns>
    internal static IReadOnlyList<IReadOnlyList<float>> CreateSamples(Random random, int valueCount)
    {
        return [.. Enumerable.Range(0, 20).Select(_ =>
            (IReadOnlyList<float>)
                [.. Enumerable.Range(0, valueCount).Select(_ => (float)((random.NextDouble() * 2) - 1))])];
    }

    /// <summary>
    /// Builds a network of every layer type and a network whose convolutional layer has several spans, each from
    /// seeded weight initializers.
    /// </summary>
    /// <remarks>
    /// Shared with the gradient clipping tests, which need parameters of every kind for the same reason.
    /// </remarks>
    /// <returns>
    /// Returns the networks, built afresh on every call.
    /// </returns>
    internal static IEnumerable<INeuralNetwork> BuildNetworks()
    {
        NeuralNetworkFactory factory = new();

        ConnectionMaskBuilder maskBuilder = new(inputCount: 5, outputCount: 3);

        maskBuilder.SetIsConnectionMasked(neuronIndex: 0, connectionIndex: 2, isMasked: true);
        maskBuilder.SetIsConnectionMasked(neuronIndex: 2, connectionIndex: 0, isMasked: true);

        yield return factory.CreateNetwork(
            factory.Concatenate(
                factory.CreateDenseLayer(3, 3, ActivationFunctions.Tanh, new GlorotUniformWeightInitializer(seed: 31)),
                factory.CreatePassthroughLayer(2)),
            factory.CreateConnectionRoutingLayer(new IndexMap([4, 3, 2, 1, 0])),
            factory.CreateSparseLayer(
                maskBuilder.Build(),
                ActivationFunctions.Eanaf,
                new GlorotUniformWeightInitializer(seed: 32)),
            factory.CreateConvolutionalLayer(
                3,
                1,
                ActivationFunctions.Softsign,
                new GlorotUniformWeightInitializer(seed: 33)));

        NeuralNetworkBuilder builder = new(inputCount: 8);

        builder.AddConvolutionalLayer(4, ActivationFunctions.Tanh, new GlorotUniformWeightInitializer(seed: 41));
        builder.AddDenseLayer(2, ActivationFunctions.Tanh, new GlorotUniformWeightInitializer(seed: 42));

        yield return builder.BuildNetwork();
    }

    #endregion
}
