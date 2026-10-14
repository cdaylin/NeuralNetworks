using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.Backpropagation.Cost;
using Daylin.NeuralNetworks.Backpropagation.LearningRate;
using Daylin.NeuralNetworks.Backpropagation.Optimization;
using Daylin.NeuralNetworks.Serialization;
using Daylin.NeuralNetworks.WeightInitialization;
using Daylin.TestUtilities;

using System.Text.Json;

namespace Daylin.NeuralNetworks.Backpropagation;

/// <summary>
/// Tests the limit on the norm of a batch's gradients.
/// </summary>
/// <remarks>
/// A network's gradients are measured rather than derived here:  one unclipped step of plain gradient descent at
/// a learning rate of one, taken on a copy, moves each parameter by exactly its mean gradient.  That works for a
/// network of any shape, and it counts a convolutional layer's shared weights once each, as the network's
/// document lists them, since that is how many times the step moves them.
/// </remarks>
[TestClass]
public class GradientClippingTests
{
    #region Private Static

    private static float LearningRate => 0.1f;

    /// <summary>
    /// A limit the one-neuron fixture's batch exceeds, so that its gradients are scaled rather than left alone.
    /// </summary>
    private static float MaxGradientNormForTheBatch => 0.1f;

    private static IReadOnlyList<IReadOnlyList<double>> BatchInput =>
        TrainingFixture.BatchInputInDoublePrecision;

    private static IReadOnlyList<double> BatchTargets => TrainingFixture.BatchTargetsInDoublePrecision;

    #endregion

    #region Public

    [TestMethod]
    public async Task AClippedBatchOfSeveralSamplesMovesAParameterByTheScaledMeanOfTheirGradientsAsync()
    {
        // Every number here comes from the reference neuron rather than from the library:  the mean over the
        //  batch's samples, the norm of those means, the factor, and the step.  Note what no clipped test can
        //  see -- once the norm exceeds the limit the step depends only on the gradients' direction, so a batch
        //  whose gradients were all averaged twice would take the same step.  That is what the unclipped
        //  BatchTrainingTests pin.
        TrainingSession session = TrainingSession.Start(
            TrainingFixture.BuildNeuronNetwork(),
            TrainingFixture.CreateSettings(
                Optimizers.GradientDescent,
                LearningRate,
                maxGradientNorm: MaxGradientNormForTheBatch));

        ReferenceNeuron neuron = TrainingFixture.CreateReferenceNeuron();

        (IReadOnlyList<double> weightGradients, double biasGradient) =
            neuron.CalculateBatchGradients(BatchInput, BatchTargets);

        double norm = Math.Sqrt(weightGradients.Sum(gradient => gradient * gradient) + (biasGradient * biasGradient));

        Assert.IsTrue(
            norm > MaxGradientNormForTheBatch,
            $"The batch's gradients do not exceed the limit ({norm}), so nothing would be scaled.");

        double scale = MaxGradientNormForTheBatch / norm;

        for (int index = 0; index < neuron.Weights.Length; index++)
            neuron.Weights[index] -= LearningRate * scale * weightGradients[index];

        neuron.Bias -= LearningRate * scale * biasGradient;

        await session.TrainBatchAsync(
            [.. BatchInput.Select(sample => (IReadOnlyList<float>)[.. sample.Select(value => (float)value)])],
            [.. BatchTargets.Select(target => (IReadOnlyList<float>)[(float)target])]);

        NeuralLayerDocument layer = TrainingFixture.ReadLayer(session.Network);

        NetworkAssert.ValuesAre(layer.WeightMatrix.Single(), neuron.Weights, "weight of");
        NetworkAssert.ValuesAre(layer.BiasVector, [neuron.Bias], "bias of");
    }

    [TestMethod]
    public async Task ANormAboveTheMaximumScalesEveryGradientDownToItAsync()
    {
        foreach (INeuralNetwork network in ReproducibilityTests.BuildNetworks())
        {
            (IReadOnlyList<IReadOnlyList<float>> input, IReadOnlyList<IReadOnlyList<float>> targetOutput) =
                CreateBatch(network);

            IReadOnlyList<double> startingParameters = ReadParameters(network);
            IReadOnlyList<double> gradients = await MeasureGradientsAsync(network, input, targetOutput);

            double norm = CalculateNorm(gradients);
            float maxGradientNorm = (float)(norm / 4);

            await Adopt(network, Optimizers.GradientDescent, maxGradientNorm).TrainBatchAsync(input, targetOutput);

            // One factor for every parameter of the network, taken from the norm of all of them together.
            double scale = maxGradientNorm / norm;

            NetworkAssert.ValuesAre(
                ReadParameters(network).Select(value => (float)value),
                startingParameters.Zip(
                    gradients,
                    (parameter, gradient) => parameter - (LearningRate * scale * gradient)),
                "clipped parameter");
        }
    }

    [TestMethod]
    public async Task ANormWithinTheMaximumLeavesTheGradientsAsTheyAreAsync()
    {
        foreach ((INeuralNetwork clipped, INeuralNetwork unclipped) in
            ReproducibilityTests.BuildNetworks().Zip(ReproducibilityTests.BuildNetworks()))
        {
            (IReadOnlyList<IReadOnlyList<float>> input, IReadOnlyList<IReadOnlyList<float>> targetOutput) =
                CreateBatch(clipped);

            double norm = CalculateNorm(await MeasureGradientsAsync(clipped, input, targetOutput));

            await Adopt(clipped, Optimizers.GradientDescent, (float)(norm * 2)).TrainBatchAsync(input, targetOutput);
            await Adopt(unclipped, Optimizers.GradientDescent, maxGradientNorm: null)
                .TrainBatchAsync(input, targetOutput);

            // Left as they are, not merely nearly so:  the same bits as training with no limit at all.
            CollectionAssert.AreEqual(
                ReadParameters(unclipped).ToList(),
                ReadParameters(clipped).ToList(),
                "Parameters trained within the limit, against parameters trained with none.");
        }
    }

    [TestMethod]
    public async Task MomentumAccumulatesTheScaledGradientsAsync()
    {
        float momentum = 0.5f;

        INeuralNetwork network = ReproducibilityTests.BuildNetworks().First();

        (IReadOnlyList<IReadOnlyList<float>> input, IReadOnlyList<IReadOnlyList<float>> targetOutput) =
            CreateBatch(network);

        IReadOnlyList<double> firstGradients = await MeasureGradientsAsync(network, input, targetOutput);
        float maxGradientNorm = (float)(CalculateNorm(firstGradients) / 3);

        TrainingSession session = Adopt(network, Optimizers.Momentum(momentum), maxGradientNorm);

        IReadOnlyList<double> startingParameters = ReadParameters(network);

        await session.TrainBatchAsync(input, targetOutput);

        double[] velocities = [.. Scale(firstGradients, maxGradientNorm)];

        IReadOnlyList<double> expectedParameters =
            [.. startingParameters.Zip(velocities, (parameter, velocity) => parameter - (LearningRate * velocity))];

        NetworkAssert.ValuesAre(
            ReadParameters(network).Select(value => (float)value),
            expectedParameters,
            "parameter after the first batch");

        // The second batch's gradients are measured where the first batch left the network, so the velocity
        // carried into it is the one the scaled first gradients built.
        IReadOnlyList<double> secondStartingParameters = ReadParameters(network);
        IReadOnlyList<double> secondGradients = await MeasureGradientsAsync(network, input, targetOutput);

        await session.TrainBatchAsync(input, targetOutput);

        IReadOnlyList<double> secondScaled = Scale(secondGradients, maxGradientNorm);

        for (int index = 0; index < velocities.Length; index++)
            velocities[index] = (momentum * velocities[index]) + secondScaled[index];

        NetworkAssert.ValuesAre(
            ReadParameters(network).Select(value => (float)value),
            secondStartingParameters.Zip(velocities, (parameter, velocity) => parameter - (LearningRate * velocity)),
            "parameter after the second batch");
    }

    [TestMethod]
    public async Task AdamSeesTheGradientsScaledRatherThanItsStepAsync()
    {
        INeuralNetwork clipped = TrainingFixture.BuildNeuronNetwork();
        INeuralNetwork unclipped = TrainingFixture.BuildNeuronNetwork();

        IReadOnlyList<IReadOnlyList<float>> input = [[0.4f, -0.6f], [-0.2f, 0.8f]];
        IReadOnlyList<IReadOnlyList<float>> targetOutput = [[1f], [-1f]];

        float maxGradientNorm = (float)(CalculateNorm(await MeasureGradientsAsync(clipped, input, targetOutput)) / 4);

        await Adopt(clipped, Optimizers.Adam(), maxGradientNorm).TrainBatchAsync(input, targetOutput);
        await Adopt(unclipped, Optimizers.Adam(), maxGradientNorm: null).TrainBatchAsync(input, targetOutput);

        // Adam's first step moves every parameter by the learning rate whatever the gradients' size, so gradients
        // scaled before Adam sees them change nothing yet.  A limit applied to the step Adam took instead would
        // have shortened it.
        NetworkAssert.ValuesAre(
            ReadParameters(clipped).Select(value => (float)value),
            ReadParameters(unclipped),
            "parameter after Adam's first step");
    }

    [TestMethod]
    public void SettingsCarryTheirMaximumThroughBeingSaved()
    {
        TrainingSettings settings = CreateSettings(Optimizers.Adam(), maxGradientNorm: 0.75f);

        TrainingSettings restored =
            JsonSerializer.Deserialize<TrainingSettings>(JsonSerializer.Serialize(settings))!;

        Assert.AreEqual(0.75f, restored.MaxGradientNorm, "Maximum gradient norm.");
    }

    [TestMethod]
    public void SettingsSavedWithoutAMaximumReadBackWithNone()
    {
        TrainingSettings settings = CreateSettings(Optimizers.Adam(), maxGradientNorm: null);

        string document = JsonSerializer.Serialize(settings);

        TrainingSettings restored = JsonSerializer.Deserialize<TrainingSettings>(document)!;

        Assert.IsNull(restored.MaxGradientNorm, "Maximum gradient norm of settings saved with none.");

        // A document written before the maximum existed carries no mention of it at all.
        string documentWithoutMaximum = document.Replace("\"MaxGradientNorm\":null,", string.Empty)
            .Replace(",\"MaxGradientNorm\":null", string.Empty);

        Assert.IsFalse(documentWithoutMaximum.Contains("MaxGradientNorm"), "The maximum, in the older document.");

        Assert.IsNull(
            JsonSerializer.Deserialize<TrainingSettings>(documentWithoutMaximum)!.MaxGradientNorm,
            "Maximum gradient norm of settings written before it existed.");
    }

    [TestMethod]
    public async Task AResumedSessionKeepsItsMaximumAsync()
    {
        TrainingSession session = TrainingSession.Start(
            TrainingFixture.BuildNeuronNetwork(),
            CreateSettings(Optimizers.Momentum(), maxGradientNorm: 0.25f));

        await session.TrainSampleAsync(TrainingFixture.BatchInput[0], TrainingFixture.BatchTargets[0]);

        TrainingSession resumed = TrainingSession.Resume(session.ToJson());

        Assert.AreEqual(0.25f, resumed.Settings.MaxGradientNorm, "Maximum gradient norm of the resumed session.");
    }

    [TestMethod]
    public void AMaximumThatIsNotPositiveIsRefused()
    {
        Asserter.AssertThat(() => CreateSettings(Optimizers.GradientDescent, maxGradientNorm: 0))
            .Throws<ArgumentOutOfRangeException>();

        Asserter.AssertThat(() => CreateSettings(Optimizers.GradientDescent, maxGradientNorm: -1))
            .Throws<ArgumentOutOfRangeException>();
    }

    #endregion

    #region Private

    private TrainingSettings CreateSettings(IOptimizer optimizer, float? maxGradientNorm) =>
        TrainingFixture.CreateSettings(optimizer, LearningRate, maxGradientNorm: maxGradientNorm);

    private TrainingSession Adopt(INeuralNetwork network, IOptimizer optimizer, float? maxGradientNorm) =>
        TrainingSession.Adopt(network, CreateSettings(optimizer, maxGradientNorm));

    /// <summary>
    /// Creates a batch of twenty samples fitting a network.
    /// </summary>
    /// <param name="network">
    /// The network the samples are for.
    /// </param>
    /// <returns>
    /// Returns the input values and the desired output values of each sample.
    /// </returns>
    private (IReadOnlyList<IReadOnlyList<float>>, IReadOnlyList<IReadOnlyList<float>>) CreateBatch(
        INeuralNetwork network)
    {
        Random random = new(7);

        return (
            ReproducibilityTests.CreateSamples(random, network.InputCount),
            ReproducibilityTests.CreateSamples(random, network.OutputCount));
    }

    /// <summary>
    /// Measures the mean gradient of every parameter of a network over a batch, leaving the network untouched.
    /// </summary>
    /// <param name="network">
    /// The network to measure.
    /// </param>
    /// <param name="input">
    /// Input values for each sample in the batch.
    /// </param>
    /// <param name="targetOutput">
    /// Desired output values for each sample in the batch.
    /// </param>
    /// <returns>
    /// Returns each parameter's mean gradient, in the order <see cref="ReadParameters"/> gives the parameters.
    /// </returns>
    private async Task<IReadOnlyList<double>> MeasureGradientsAsync(
        INeuralNetwork network,
        IReadOnlyList<IReadOnlyList<float>> input,
        IReadOnlyList<IReadOnlyList<float>> targetOutput)
    {
        TrainingSession measuring = TrainingSession.Start(
            network,
            new TrainingSettings(
                CostFunctions.MeanSquaredError,
                Optimizers.GradientDescent,
                LearningRateSchedulers.ConstantLearningRateScheduler(1)));

        await measuring.TrainBatchAsync(input, targetOutput);

        return [.. ReadParameters(network).Zip(ReadParameters(measuring.Network), (before, after) => before - after)];
    }

    /// <summary>
    /// Reads every weight and bias of a network, each parameter once, in a fixed order.
    /// </summary>
    /// <param name="network">
    /// The network to read.
    /// </param>
    /// <returns>
    /// Returns the parameters, layer by layer, each layer's weights before its biases.
    /// </returns>
    private IReadOnlyList<double> ReadParameters(INeuralNetwork network)
    {
        List<double> parameters = [];

        foreach (NeuralLayerDocument layer in NetworkDocument.FromNetwork(network).NeuralLayers)
        {
            parameters.AddRange(layer.WeightMatrix.SelectMany(row => row).Select(weight => (double)weight));
            parameters.AddRange(layer.BiasVector.Select(bias => (double)bias));
        }

        return parameters;
    }

    private double CalculateNorm(IEnumerable<double> gradients) =>
        Math.Sqrt(gradients.Sum(gradient => gradient * gradient));

    /// <summary>
    /// Scales gradients down to a maximum norm, as the settings describe, where their norm exceeds it.
    /// </summary>
    /// <param name="gradients">
    /// Every parameter's gradient.
    /// </param>
    /// <param name="maxGradientNorm">
    /// The largest the norm may be.
    /// </param>
    /// <returns>
    /// Returns the gradients, scaled where they needed to be.
    /// </returns>
    private IReadOnlyList<double> Scale(IReadOnlyList<double> gradients, float maxGradientNorm)
    {
        double norm = CalculateNorm(gradients);

        double scale = norm > maxGradientNorm ? maxGradientNorm / norm : 1;

        return [.. gradients.Select(gradient => gradient * scale)];
    }

    #endregion
}
