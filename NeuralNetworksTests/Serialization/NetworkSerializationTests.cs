using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.Backpropagation;
using Daylin.NeuralNetworks.Backpropagation.LearningRate;
using Daylin.NeuralNetworks.Backpropagation.Optimization;
using Daylin.NeuralNetworks.LayerTemplates;
using Daylin.NeuralNetworks.WeightInitialization;

namespace Daylin.NeuralNetworks.Serialization;

/// <summary>
/// Tests that a neural network survives being saved and restored.
/// </summary>
/// <remarks>
/// A network is persisted as JSON, which is how a trained network outlives the process that trained it.  What a
/// restored network owes its caller is everything the saved one could do:  the same output for the same input,
/// the same weights, and training that continues rather than restarts.
/// </remarks>
[TestClass]
public class NetworkSerializationTests
{
    #region Private Static

    private static IEnumerable<float> Input => [0.25f, -0.5f, 0.75f, 1f];

    #endregion

    #region Public

    [TestMethod]
    public async Task ARestoredNetworkCalculatesWhatTheSavedNetworkCalculatedAsync()
    {
        INeuralNetwork network = BuildNetworkOfEveryLayerType();

        INeuralNetwork restored = NetworkDocument.FromJson(NetworkDocument.FromNetwork(network).ToJson())
            .BuildNetwork();

        Assert.AreEqual(network.ShortDescription, restored.ShortDescription);
        Assert.AreEqual(network.InputCount, restored.InputCount);
        Assert.AreEqual(network.OutputCount, restored.OutputCount);

        IEnumerable<float> expectedOutput = await network.CalculateOutputAsync(Input);

        NetworkAssert.ValuesAre(
            await restored.CalculateOutputAsync(Input),
            expectedOutput.Select(value => (double)value),
            "restored network's output");
    }

    [TestMethod]
    public void ARestoredNetworkCarriesTheWeightsItWasSavedWith()
    {
        NetworkDocument saved = NetworkDocument.FromNetwork(BuildNetworkOfEveryLayerType());

        NetworkDocument restored = NetworkDocument.FromNetwork(saved.BuildNetwork());

        Assert.AreEqual(saved.NeuralLayers.Count, restored.NeuralLayers.Count, "Count of neural layers.");

        for (int layerIndex = 0; layerIndex < saved.NeuralLayers.Count; layerIndex++)
        {
            NeuralLayerDocument savedLayer = saved.NeuralLayers[layerIndex];
            NeuralLayerDocument restoredLayer = restored.NeuralLayers[layerIndex];

            Assert.AreEqual(savedLayer.LayerTypeName, restoredLayer.LayerTypeName);
            Assert.IsTrue(savedLayer.BiasVector.SequenceEqual(restoredLayer.BiasVector), "Biases.");

            Assert.IsTrue(
                savedLayer.WeightMatrix.Zip(restoredLayer.WeightMatrix)
                    .All(rows => rows.First.SequenceEqual(rows.Second)),
                $"Weights of layer {layerIndex}.");
        }
    }

    [TestMethod]
    public async Task ARestoredNetworkContinuesTrainingWhereTheSavedNetworkLeftOffAsync()
    {
        INeuralNetwork network = BuildNetworkOfEveryLayerType();

        TrainingSettings settings = new(
            CostDerivatives.MeanSquaredError,
            Optimizers.GradientDescent(),
            LearningRateSchedulers.ConstantLearningRateScheduler(0.5f));

        await TrainingSession.Adopt(network, settings).TrainSampleAsync([.. Input], [1f]);

        INeuralNetwork restored = NetworkDocument.FromNetwork(network).BuildNetwork();

        await TrainingSession.Adopt(network, settings).TrainSampleAsync([.. Input], [1f]);
        await TrainingSession.Adopt(restored, settings).TrainSampleAsync([.. Input], [1f]);

        IEnumerable<float> expectedOutput = await network.CalculateOutputAsync(Input);

        NetworkAssert.ValuesAre(
            await restored.CalculateOutputAsync(Input),
            expectedOutput.Select(value => (double)value),
            "further trained network's output");
    }

    [TestMethod]
    public async Task ASavedNetworkCarriesNoTrainingStateAsync()
    {
        TrainingSession session = TrainingSession.Start(
            BuildNetworkOfEveryLayerType(),
            new TrainingSettings(
                CostDerivatives.MeanSquaredError,
                Optimizers.Adam(),
                LearningRateSchedulers.ConstantLearningRateScheduler(0.1f)));

        await TrainBatchesAsync(session, batchCount: 2);

        // A network is what calculates output; how it came to hold its weights belongs to the session.
        string document = NetworkDocument.FromNetwork(session.Network).ToJson();

        Assert.IsFalse(document.Contains("Adam", StringComparison.Ordinal), "Optimizer state in the document.");
        Assert.IsFalse(document.Contains("Epoch", StringComparison.Ordinal), "Epoch count in the document.");
    }

    [TestMethod]
    public async Task AResumedSessionCarriesTheStateItsOptimizerHadReachedAsync()
    {
        TrainingSession session = TrainingSession.Start(
            BuildNetworkOfEveryLayerType(),
            new TrainingSettings(
                CostDerivatives.MeanSquaredError,
                Optimizers.Adam(),
                LearningRateSchedulers.ConstantLearningRateScheduler(0.1f)));

        await TrainBatchesAsync(session, batchCount: 2);

        TrainingSession resumed = TrainingSession.Resume(session.ToJson());

        // Adam's step depends on the moments its earlier steps left behind, so a session that had lost them
        // would move differently from here on.
        await TrainBatchesAsync(session, batchCount: 2);
        await TrainBatchesAsync(resumed, batchCount: 2);

        IEnumerable<float> expectedOutput = await session.Network.CalculateOutputAsync(Input);

        NetworkAssert.ValuesAre(
            await resumed.Network.CalculateOutputAsync(Input),
            expectedOutput.Select(value => (double)value),
            "further trained network's output");
    }

    [TestMethod]
    public async Task AResumedSessionHasCompletedTheEpochsTheSavedOneHadAsync()
    {
        TrainingSession session = TrainingSession.Start(
            BuildNetworkOfEveryLayerType(),
            new TrainingSettings(
                CostDerivatives.MeanSquaredError,
                Optimizers.GradientDescent(),
                LearningRateSchedulers.ExponentialDecayLearningRateScheduler(0.5f, decayRate: 0.5f)));

        await session.TrainEpochAsync([[.. Input], [.. Input]], [[1f], [-1f]], batchSize: 2);
        await session.TrainEpochAsync([[.. Input], [.. Input]], [[1f], [-1f]], batchSize: 2);

        TrainingSession resumed = TrainingSession.Resume(session.ToJson());

        Assert.AreEqual(session.CompletedEpochCount, resumed.CompletedEpochCount, "Completed epochs.");

        // The count is what the schedule reduces the rate over, so a session that had lost it would train its
        // next epoch at a rate two epochs too high.
        await session.TrainEpochAsync([[.. Input], [.. Input]], [[1f], [-1f]], batchSize: 2);
        await resumed.TrainEpochAsync([[.. Input], [.. Input]], [[1f], [-1f]], batchSize: 2);

        IEnumerable<float> expectedOutput = await session.Network.CalculateOutputAsync(Input);

        NetworkAssert.ValuesAre(
            await resumed.Network.CalculateOutputAsync(Input),
            expectedOutput.Select(value => (double)value),
            "further trained network's output");
    }

    [TestMethod]
    public async Task AResumedSessionKeepsTrainingTheNetworkItWasSavedWithAsync()
    {
        TrainingSession session = TrainingSession.Start(
            BuildNetworkOfEveryLayerType(),
            new TrainingSettings(
                CostDerivatives.MeanSquaredError,
                Optimizers.Momentum(),
                LearningRateSchedulers.ConstantLearningRateScheduler(0.1f)));

        await TrainBatchesAsync(session, batchCount: 2);

        TrainingSession resumed = TrainingSession.Resume(session.ToJson());

        IEnumerable<float> expectedOutput = await session.Network.CalculateOutputAsync(Input);

        NetworkAssert.ValuesAre(
            await resumed.Network.CalculateOutputAsync(Input),
            expectedOutput.Select(value => (double)value),
            "resumed session's network output");
    }

    [TestMethod]
    public async Task ADocumentCarryingNeitherOptimizerStateNorEpochCountStillLoadsAsync()
    {
        // A document a network was saved to before either was recorded, and before a weight initializer
        // recorded its seed.  Reading one is what a saved network's owner is owed.
        string document = """
            {
              "$type": "NeuralNetwork",
              "NetworkLayer": {
                "$type": "ConvolutionalLayer",
                "WeightVector": [0.5, -0.25],
                "Bias": 0.1,
                "InputCount": 4,
                "OutputCount": 2,
                "ActivationFunction": { "$type": "Tanh" },
                "WeightInitializer": { "$type": "GlorotUniform", "MaxRange": 1 }
              }
            }
            """;

        INeuralNetwork network = NetworkDocument.FromJson(document).BuildNetwork();

        await NetworkAssert.OutputIsAsync(
            network,
            [0.4f, -0.6f, 0.8f, 1f],
            Math.Tanh((0.5 * 0.4) + (-0.25 * -0.6) + 0.1),
            Math.Tanh((0.5 * 0.8) + (-0.25 * 1.0) + 0.1));
    }

    [TestMethod]
    public void ARestoredNetworkKeepsTheSeedItsWeightsWereGeneratedFrom()
    {
        NetworkDocument saved = NetworkDocument.FromNetwork(BuildNetworkOfEveryLayerType());

        NetworkDocument restored = NetworkDocument.FromNetwork(saved.BuildNetwork());

        foreach (NeuralLayerDocument layer in restored.NeuralLayers)
            Assert.AreEqual(Seed, layer.WeightInitializerSeed, $"Seed of the {layer.LayerTypeName}.");
    }

    #endregion

    #region Private

    private int Seed => 1;

    /// <summary>
    /// Trains a session on the same two-sample batch several times over.
    /// </summary>
    /// <param name="session">
    /// The session to train.
    /// </param>
    /// <param name="batchCount">
    /// Number of batches to train on.
    /// </param>
    /// <returns>
    /// Returns a task representing this <c>async</c> operation.
    /// </returns>
    private async Task TrainBatchesAsync(TrainingSession session, int batchCount)
    {
        for (int batchIndex = 0; batchIndex < batchCount; batchIndex++)
            await session.TrainBatchAsync([[.. Input], [.. Input.Reverse()]], [[1f], [-1f]]);
    }

    private IConnectionWeightInitializer WeightInitializer =>
        new GlorotUniformWeightInitializer(seed: Seed);

    /// <summary>
    /// Builds a network holding one layer of every type the library provides.
    /// </summary>
    /// <returns>
    /// Returns a network of four input values and one output value.
    /// </returns>
    private INeuralNetwork BuildNetworkOfEveryLayerType()
    {
        NeuralNetworkFactory factory = new();

        INetworkLayerTemplate head = factory.Concatenate(
            factory.CreateDenseLayer(2, 2, ActivationFunctions.Tanh, WeightInitializer),
            factory.CreatePassthroughLayer(2));

        ConnectionMaskBuilder maskBuilder = new(inputCount: 4, outputCount: 2);

        maskBuilder.SetIsConnectionMasked(neuronIndex: 0, connectionIndex: 3, isMasked: true);

        return factory.CreateNetwork(
            head,
            factory.CreateConnectionRoutingLayer(new IndexMap([3, 2, 1, 0])),
            factory.CreateSparseLayer(maskBuilder.Build(), ActivationFunctions.Tanh, WeightInitializer),
            factory.CreateConvolutionalLayer(2, 1, ActivationFunctions.Tanh, WeightInitializer));
    }

    #endregion
}
