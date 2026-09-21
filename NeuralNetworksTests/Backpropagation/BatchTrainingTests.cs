using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.Backpropagation.LearningRate;
using Daylin.NeuralNetworks.Backpropagation.Optimization;
using Daylin.NeuralNetworks.Serialization;
using Daylin.NeuralNetworks.WeightInitialization;

namespace Daylin.NeuralNetworks.Backpropagation;

/// <summary>
/// Tests training a neural network on batches of samples, through a training session.
/// </summary>
/// <remarks>
/// A batch moves a network's parameters once, by what its samples call for averaged over the batch, so every
/// sample in a batch is learned from as the batch found the network rather than as the sample before it left
/// it.  The network a session trains is the session's own copy, so these tests read their results from
/// <see cref="TrainingSession.Network"/>.
/// </remarks>
[TestClass]
public class BatchTrainingTests
{
    #region Private Static

    private static float LearningRate => 0.5f;

    private static IReadOnlyList<IReadOnlyList<float>> BatchInput { get; } =
        [[0.4f, -0.6f], [-0.2f, 0.8f], [1f, 1f]];

    private static IReadOnlyList<IReadOnlyList<float>> BatchTargets { get; } = [[1f], [-1f], [0.5f]];

    #endregion

    #region Public

    [TestMethod]
    public async Task OneSampleIsABatchOfOneAsync()
    {
        TrainingSession trainedOnOneSample = StartSession(Optimizers.GradientDescent());
        TrainingSession trainedOnABatch = StartSession(Optimizers.GradientDescent());

        await trainedOnOneSample.TrainSampleAsync([.. BatchInput[0]], [.. BatchTargets[0]]);

        await trainedOnABatch.TrainBatchAsync([BatchInput[0]], [BatchTargets[0]]);

        AssertNetworksAgree(trainedOnOneSample.Network, trainedOnABatch.Network);
    }

    [TestMethod]
    public async Task ABatchMovesEachParameterByWhatItsSamplesCallForAveragedOverThemAsync()
    {
        TrainingSession session = StartSession(Optimizers.GradientDescent());

        await session.TrainBatchAsync(BatchInput, BatchTargets);

        ReferenceNeuron neuron = CreateReferenceNeuron();

        (IReadOnlyList<double> weightGradients, double biasGradient) = neuron.CalculateBatchGradients(
            ToDouble(BatchInput),
            BatchTargets.Select(target => (double)target[0]).ToList());

        NeuralLayerDocument layer = ReadLayer(session.Network);

        NetworkAssert.ValuesAre(
            layer.WeightMatrix.Single(),
            neuron.Weights.Zip(weightGradients, (weight, gradient) => weight - (LearningRate * gradient)),
            "weight of");

        NetworkAssert.ValuesAre(
            layer.BiasVector,
            [neuron.Bias - (LearningRate * biasGradient)],
            "bias of");
    }

    [TestMethod]
    public async Task ABatchMovesTheNetworkTheSameWayHoweverItsSamplesAreOrderedAsync()
    {
        TrainingSession session = StartSession(Optimizers.GradientDescent());
        TrainingSession reversedSession = StartSession(Optimizers.GradientDescent());

        await session.TrainBatchAsync(BatchInput, BatchTargets);

        await reversedSession.TrainBatchAsync([.. BatchInput.Reverse()], [.. BatchTargets.Reverse()]);

        AssertNetworksAgree(session.Network, reversedSession.Network);
    }

    [TestMethod]
    public async Task StartingASessionLeavesTheNetworkItWasGivenUntouchedAsync()
    {
        INeuralNetwork network = BuildNetwork();

        NeuralLayerDocument layerBeforeTraining = ReadLayer(network);

        TrainingSession session = TrainingSession.Start(network, CreateSettings(Optimizers.GradientDescent()));

        await session.TrainBatchAsync(BatchInput, BatchTargets);

        AssertLayerIsUnmoved(network, layerBeforeTraining);

        // The session's own network is what moved.
        Assert.IsFalse(
            ReadLayer(session.Network).BiasVector.SequenceEqual(layerBeforeTraining.BiasVector),
            "Biases of the network the session trained.");
    }

    [TestMethod]
    public async Task AnAdoptedNetworkIsTheOneTheSessionTrainsAsync()
    {
        INeuralNetwork network = BuildNetwork();

        NeuralLayerDocument layerBeforeTraining = ReadLayer(network);

        TrainingSession session = TrainingSession.Adopt(network, CreateSettings(Optimizers.GradientDescent()));

        await session.TrainBatchAsync(BatchInput, BatchTargets);

        // The caller's own network moved, which is the whole difference from Start.
        Assert.IsFalse(
            ReadLayer(network).BiasVector.SequenceEqual(layerBeforeTraining.BiasVector),
            "Biases of the adopted network.");

        AssertNetworksAgree(network, session.Network);
    }

    [TestMethod]
    public async Task TwoSessionsStartedFromOneNetworkDoNotDisturbEachOtherAsync()
    {
        INeuralNetwork network = BuildNetwork();

        TrainingSession session = TrainingSession.Start(network, CreateSettings(Optimizers.Momentum()));
        TrainingSession otherSession = TrainingSession.Start(network, CreateSettings(Optimizers.Momentum()));

        // One session trains twice as much as the other; neither sees the other's weights or velocities.
        await session.TrainBatchAsync(BatchInput, BatchTargets);
        await session.TrainBatchAsync(BatchInput, BatchTargets);
        await otherSession.TrainBatchAsync(BatchInput, BatchTargets);

        TrainingSession trainedOnceAlone = TrainingSession.Start(network, CreateSettings(Optimizers.Momentum()));

        await trainedOnceAlone.TrainBatchAsync(BatchInput, BatchTargets);

        AssertNetworksAgree(trainedOnceAlone.Network, otherSession.Network);
    }

    [TestMethod]
    public async Task ABatchAbandonedPartwayThroughLeavesTheNetworkAsItFoundItAsync()
    {
        using CancellationTokenSource cancellation = new();

        // Cancelled once the batch has learned from a sample, so that there are accumulated gradients for the
        // network to discard rather than none.
        TrainingSession session = StartSession(
            Optimizers.GradientDescent(),
            costDerivative: new CancellingCostDerivative(cancellation));

        NeuralLayerDocument layerBeforeTraining = ReadLayer(session.Network);

        await Assert.ThrowsExactlyAsync<OperationCanceledException>(
            () => session.TrainBatchAsync(BatchInput, BatchTargets, cancellation.Token));

        AssertLayerIsUnmoved(session.Network, layerBeforeTraining);
    }

    [TestMethod]
    public async Task GradientsLeftByAnAbandonedBatchDoNotReachTheNextOneAsync()
    {
        using CancellationTokenSource cancellation = new();

        TrainingSession abandonedFirst = StartSession(
            Optimizers.GradientDescent(),
            costDerivative: new CancellingCostDerivative(cancellation));

        await Assert.ThrowsExactlyAsync<OperationCanceledException>(
            () => abandonedFirst.TrainBatchAsync(BatchInput, BatchTargets, cancellation.Token));

        // A second session over the same batch, with nothing abandoned before it.
        TrainingSession trainedOnce = StartSession(Optimizers.GradientDescent());

        await abandonedFirst.TrainBatchAsync(BatchInput, BatchTargets);
        await trainedOnce.TrainBatchAsync(BatchInput, BatchTargets);

        AssertNetworksAgree(abandonedFirst.Network, trainedOnce.Network);
    }

    [TestMethod]
    public async Task AConvolutionalLayersSharedWeightsGatherEverySpanOfEverySampleAsync()
    {
        NeuralNetworkBuilder builder = new(inputCount: 4);

        builder.AddConvolutionalLayer(2, ActivationFunctions.Tanh, WeightInitializer);

        NetworkDocument document = NetworkDocument.FromNetwork(builder.BuildNetwork());

        document.NeuralLayers[0].SetWeightMatrix([0.5f, -0.25f]);
        document.NeuralLayers[0].SetBiasVector(0.1f);

        TrainingSession session =
            TrainingSession.Start(document.BuildNetwork(), CreateSettings(Optimizers.GradientDescent()));

        IReadOnlyList<IReadOnlyList<float>> input = [[0.4f, -0.6f, 0.8f, 1f], [-0.2f, 0.8f, 0.1f, -0.3f]];
        IReadOnlyList<IReadOnlyList<float>> targets = [[1f, -1f], [-1f, 0.5f]];

        await session.TrainBatchAsync(input, targets);

        // One neuron stands for the shared weights:  each span of each sample is a sample of its own, and the
        // batch averages over the samples rather than over the spans, so four spans over two samples average
        // to the sum of each sample's two spans, halved.
        ReferenceNeuron neuron = new([0.5, -0.25], 0.1);

        double[] weightGradients = new double[2];
        double biasGradient = 0;

        for (int sampleIndex = 0; sampleIndex < input.Count; sampleIndex++)
        {
            for (int spanIndex = 0; spanIndex < 2; spanIndex++)
            {
                IReadOnlyList<double> span =
                    [input[sampleIndex][spanIndex * 2], input[sampleIndex][(spanIndex * 2) + 1]];

                (IReadOnlyList<double> spanWeights, double spanBias) =
                    neuron.CalculateGradients(span, targets[sampleIndex][spanIndex]);

                for (int index = 0; index < weightGradients.Length; index++)
                    weightGradients[index] += spanWeights[index] / input.Count;

                biasGradient += spanBias / input.Count;
            }
        }

        NeuralLayerDocument layer = ReadLayer(session.Network);

        NetworkAssert.ValuesAre(
            layer.WeightMatrix.Single(),
            neuron.Weights.Zip(weightGradients, (weight, gradient) => weight - (LearningRate * gradient)),
            "shared weight of");

        NetworkAssert.ValuesAre(layer.BiasVector, [0.1 - (LearningRate * biasGradient)], "shared bias of");
    }

    [TestMethod]
    public async Task AnEpochTrainsEveryBatchAndTheRemainderThatFollowsThemAsync()
    {
        TrainingSession trainedByEpoch = StartSession(Optimizers.GradientDescent());
        TrainingSession trainedByBatch = StartSession(Optimizers.GradientDescent());

        await trainedByEpoch.TrainEpochAsync(BatchInput, BatchTargets, batchSize: 2);

        // Three samples in batches of two is a batch of two and a batch of one.
        await trainedByBatch.TrainBatchAsync(
            [BatchInput[0], BatchInput[1]],
            [BatchTargets[0], BatchTargets[1]]);

        await trainedByBatch.TrainBatchAsync([BatchInput[2]], [BatchTargets[2]]);

        AssertNetworksAgree(trainedByEpoch.Network, trainedByBatch.Network);
    }

    [TestMethod]
    public async Task AnEpochIsCountedOnceItsLastBatchIsDoneAsync()
    {
        TrainingSession session = StartSession(Optimizers.GradientDescent());

        Assert.AreEqual(0, session.CompletedEpochCount, "Epochs completed by a session that has not trained.");

        await session.TrainBatchAsync(BatchInput, BatchTargets);

        Assert.AreEqual(0, session.CompletedEpochCount, "Epochs completed by training on a batch.");

        await session.TrainEpochAsync(BatchInput, BatchTargets, batchSize: 2);
        await session.TrainEpochAsync(BatchInput, BatchTargets, batchSize: 2);

        Assert.AreEqual(2, session.CompletedEpochCount, "Epochs completed by training on a set of samples.");
    }

    [TestMethod]
    public async Task EveryBatchOfAnEpochTrainsAtTheEpochsLearningRateAsync()
    {
        TrainingSettings decaying = new(
            CostDerivatives.MeanSquaredError,
            Optimizers.GradientDescent(),
            LearningRateSchedulers.ExponentialDecayLearningRateScheduler(LearningRate, decayRate: 0.5f));

        TrainingSession trainedByEpoch = TrainingSession.Start(BuildNetwork(), decaying);
        TrainingSession trainedByBatch = StartSession(Optimizers.GradientDescent());

        await trainedByEpoch.TrainEpochAsync(BatchInput, BatchTargets, batchSize: 2);

        // Both batches of the first epoch train at the initial rate.
        await trainedByBatch.TrainBatchAsync(
            [BatchInput[0], BatchInput[1]],
            [BatchTargets[0], BatchTargets[1]]);

        await trainedByBatch.TrainBatchAsync([BatchInput[2]], [BatchTargets[2]]);

        AssertNetworksAgree(trainedByEpoch.Network, trainedByBatch.Network);

        // The second epoch's batches train at half of it, which a session started on that rate matches.
        TrainingSession trainedAtHalfTheRate = TrainingSession.Start(
            trainedByBatch.Network,
            CreateSettings(Optimizers.GradientDescent(), LearningRate / 2));

        await trainedByEpoch.TrainEpochAsync(BatchInput, BatchTargets, batchSize: 2);

        await trainedAtHalfTheRate.TrainBatchAsync(
            [BatchInput[0], BatchInput[1]],
            [BatchTargets[0], BatchTargets[1]]);

        await trainedAtHalfTheRate.TrainBatchAsync([BatchInput[2]], [BatchTargets[2]]);

        AssertNetworksAgree(trainedByEpoch.Network, trainedAtHalfTheRate.Network);
    }

    [TestMethod]
    public async Task TrainingRejectsABatchWhoseInputsAndTargetsDoNotPairUpAsync()
    {
        TrainingSession session = StartSession(Optimizers.GradientDescent());

        await Assert.ThrowsExactlyAsync<ArgumentOutOfRangeException>(
            () => session.TrainBatchAsync(BatchInput, [BatchTargets[0]]));

        await Assert.ThrowsExactlyAsync<ArgumentException>(
            () => session.TrainBatchAsync([], []));
    }

    #endregion

    #region Private

    private IConnectionWeightInitializer WeightInitializer { get; } = new GlorotUniformWeightInitializer(seed: 1);

    /// <summary>
    /// Builds a network of one neuron over two input values, with known weights and bias.
    /// </summary>
    /// <returns>
    /// Returns the network, which <see cref="CreateReferenceNeuron"/> describes.
    /// </returns>
    private INeuralNetwork BuildNetwork()
    {
        NeuralNetworkBuilder builder = new(inputCount: 2);

        builder.AddDenseLayer(1, ActivationFunctions.Tanh, WeightInitializer);

        NetworkDocument document = NetworkDocument.FromNetwork(builder.BuildNetwork());

        document.NeuralLayers[0].SetWeightMatrix([0.5f, -0.25f]);
        document.NeuralLayers[0].SetBiasVector(0.1f);

        return document.BuildNetwork();
    }

    private ReferenceNeuron CreateReferenceNeuron() => new([0.5, -0.25], 0.1);

    private TrainingSettings CreateSettings(
        IOptimizer optimizer,
        float learningRate = 0.5f,
        ICostDerivative? costDerivative = null)
    {
        return new TrainingSettings(
            costDerivative ?? CostDerivatives.MeanSquaredError,
            optimizer,
            LearningRateSchedulers.ConstantLearningRateScheduler(learningRate));
    }

    private TrainingSession StartSession(
        IOptimizer optimizer,
        float learningRate = 0.5f,
        ICostDerivative? costDerivative = null)
    {
        return TrainingSession.Start(BuildNetwork(), CreateSettings(optimizer, learningRate, costDerivative));
    }

    private NeuralLayerDocument ReadLayer(INeuralNetwork network) =>
        NetworkDocument.FromNetwork(network).NeuralLayers.Single();

    private IReadOnlyList<IReadOnlyList<double>> ToDouble(IReadOnlyList<IReadOnlyList<float>> samples) =>
        samples.Select(sample => (IReadOnlyList<double>)sample.Select(value => (double)value).ToList()).ToList();

    private void AssertLayerIsUnmoved(INeuralNetwork network, NeuralLayerDocument layerBeforeTraining)
    {
        NeuralLayerDocument layer = ReadLayer(network);

        NetworkAssert.ValuesAre(
            layer.WeightMatrix.Single(),
            layerBeforeTraining.WeightMatrix.Single().Select(weight => (double)weight),
            "weight of");

        NetworkAssert.ValuesAre(
            layer.BiasVector,
            layerBeforeTraining.BiasVector.Select(bias => (double)bias),
            "bias of");
    }

    private void AssertNetworksAgree(INeuralNetwork network, INeuralNetwork other)
    {
        NeuralLayerDocument layer = ReadLayer(network);
        NeuralLayerDocument otherLayer = ReadLayer(other);

        for (int rowIndex = 0; rowIndex < layer.WeightMatrix.Count; rowIndex++)
        {
            NetworkAssert.ValuesAre(
                otherLayer.WeightMatrix[rowIndex],
                layer.WeightMatrix[rowIndex].Select(weight => (double)weight),
                $"weight in row {rowIndex} of");
        }

        NetworkAssert.ValuesAre(
            otherLayer.BiasVector,
            layer.BiasVector.Select(bias => (double)bias),
            "bias of");
    }

    /// <summary>
    /// A cost derivative that cancels a batch once it has learned from its first sample.
    /// </summary>
    /// <remarks>
    /// It is called once for each of a sample's output values, and the networks these tests build have one, so
    /// the second call belongs to the second sample:  by then the first sample's gradients have been
    /// accumulated and are the network's to discard.
    /// </remarks>
    private sealed class CancellingCostDerivative : ICostDerivative
    {
        public CancellingCostDerivative(CancellationTokenSource cancellation)
        {
            cancellation.ThrowIfNull();

            Cancellation = cancellation;
        }

        public string ShortName => "CancelsAfterOneSample";

        public float Calculate(float targetValue, float actualValue)
        {
            CallCount += 1;

            if (CallCount == 2)
                Cancellation.Cancel();

            return CostDerivatives.MeanSquaredError.Calculate(targetValue, actualValue);
        }

        private CancellationTokenSource Cancellation { get; }

        private int CallCount { get; set; }
    }

    #endregion
}
