using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.Backpropagation.Cost;
using Daylin.NeuralNetworks.Backpropagation.LearningRate;
using Daylin.NeuralNetworks.Backpropagation.Optimization;
using Daylin.NeuralNetworks.Serialization;
using Daylin.NeuralNetworks.WeightInitialization;

namespace Daylin.NeuralNetworks.Backpropagation;

/// <summary>
/// The network, the settings and the assertions the training tests share.
/// </summary>
/// <remarks>
/// One neuron over two input values, with weights and a bias written in rather than initialized, so that every
/// expected value can be computed by hand:  <see cref="CreateReferenceNeuron"/> is the same neuron in double
/// precision.  Several test classes train this same network, so it is built here rather than in each of them,
/// where the three numbers would have to agree in five places.
/// </remarks>
internal static class TrainingFixture
{
    #region Public Static

    /// <summary>
    /// The learning rate the training tests use unless they say otherwise.
    /// </summary>
    /// <remarks>
    /// A constant rather than the expression-bodied property the house style prefers, so that it can be the
    /// default value of the parameters below:  the rate a test computes against and the rate its session trains
    /// at are then one number.
    /// </remarks>
    public const float LearningRate = 0.5f;

    /// <summary>
    /// The input values the training tests feed the one-neuron network.
    /// </summary>
    public static IReadOnlyList<float> Input { get; } = [0.3f, -0.7f];

    /// <summary>
    /// The output value the training tests train that network towards.
    /// </summary>
    public static IReadOnlyList<float> Target { get; } = [0.25f];

    /// <summary>
    /// The two samples the batch tests feed the one-neuron network.
    /// </summary>
    /// <remarks>
    /// Kept beside <see cref="BatchInputInDoublePrecision"/> and the targets, since a test that trains on one
    /// and computes its expected values from the other has to be reading the same numbers twice.
    /// </remarks>
    public static IReadOnlyList<IReadOnlyList<float>> BatchInput { get; } = [[0.4f, -0.6f], [-0.2f, 0.8f]];

    /// <summary>
    /// The output value each of those samples trains towards, as the training methods take them.
    /// </summary>
    public static IReadOnlyList<IReadOnlyList<float>> BatchTargets { get; } = [[1f], [-1f]];

    /// <summary>
    /// The same two samples, for the reference neuron that computes expected values in double precision.
    /// </summary>
    public static IReadOnlyList<IReadOnlyList<double>> BatchInputInDoublePrecision { get; } =
        [[0.4, -0.6], [-0.2, 0.8]];

    /// <summary>
    /// The same two targets, for the reference neuron.
    /// </summary>
    public static IReadOnlyList<double> BatchTargetsInDoublePrecision { get; } = [1, -1];

    /// <summary>
    /// The weights the one neuron starts with.
    /// </summary>
    public static IReadOnlyList<double> StartingWeights { get; } = [0.5, -0.25];

    /// <summary>
    /// The bias the one neuron starts with.
    /// </summary>
    public static double StartingBias => 0.1;

    /// <summary>
    /// Builds a network of one neuron, with the fixture's known weights and bias over two input values, or
    /// initializer-drawn weights over any other number.
    /// </summary>
    /// <remarks>
    /// Only the default argument gives weights <see cref="CreateReferenceNeuron"/> describes, so a test that
    /// computes expected values against the reference neuron must take it.  Another input count is for a test
    /// that needs a network of another shape and does not care what it holds.
    /// </remarks>
    /// <param name="inputCount">
    /// Number of input values the neuron takes;  the fixture's weights describe two of them.
    /// </param>
    /// <returns>
    /// Returns the network, which <see cref="CreateReferenceNeuron"/> describes for two input values.
    /// </returns>
    public static INeuralNetwork BuildNeuronNetwork(int inputCount = 2)
    {
        NeuralNetworkBuilder builder = new(inputCount);

        builder.AddDenseLayer(1, ActivationFunctions.Tanh, new GlorotUniformWeightInitializer(seed: 1));

        if (inputCount != StartingWeights.Count)
            return builder.BuildNetwork();

        NetworkDocument document = NetworkDocument.FromNetwork(builder.BuildNetwork());

        document.NeuralLayers[0].SetWeightMatrix([(float)StartingWeights[0], (float)StartingWeights[1]]);
        document.NeuralLayers[0].SetBiasVector((float)StartingBias);

        return document.BuildNetwork();
    }

    /// <summary>
    /// Creates the reference neuron describing the network <see cref="BuildNeuronNetwork"/> builds.
    /// </summary>
    /// <remarks>
    /// The same weights and bias the network is built with, in double precision:  a test computes what training
    /// should leave from this and compares the network against it, so the two cannot drift apart.
    /// </remarks>
    /// <returns>
    /// Returns a reference neuron holding the same weights and bias, in double precision.
    /// </returns>
    public static ReferenceNeuron CreateReferenceNeuron() => new(StartingWeights, StartingBias);

    /// <summary>
    /// Builds a network of two dense layers, so that a test can abandon a batch partway through a layer tree
    /// deep enough for one layer's gradients to reach another.
    /// </summary>
    /// <returns>
    /// Returns the network:  two input values, two neurons, then one.
    /// </returns>
    public static INeuralNetwork BuildTwoLayerNetwork()
    {
        NeuralNetworkBuilder builder = new(inputCount: 2);

        builder.AddDenseLayer(2, ActivationFunctions.Tanh, new GlorotUniformWeightInitializer(seed: 21));
        builder.AddDenseLayer(1, ActivationFunctions.Tanh, new GlorotUniformWeightInitializer(seed: 22));

        return builder.BuildNetwork();
    }

    /// <summary>
    /// Creates training settings.
    /// </summary>
    /// <param name="optimizer">
    /// The optimizer to train with.
    /// </param>
    /// <param name="learningRate">
    /// The constant learning rate to train at.
    /// </param>
    /// <param name="costFunction">
    /// The cost function to train against, or <see langword="null"/> for mean squared error.
    /// </param>
    /// <param name="maxGradientNorm">
    /// The largest the norm of a batch's gradients may be, or <see langword="null"/> for no limit.
    /// </param>
    /// <returns>
    /// Returns the settings.
    /// </returns>
    public static TrainingSettings CreateSettings(
        IOptimizer optimizer,
        float learningRate = LearningRate,
        ICostFunction? costFunction = null,
        float? maxGradientNorm = null)
    {
        return new TrainingSettings(
            costFunction ?? CostFunctions.MeanSquaredError,
            optimizer,
            LearningRateSchedulers.ConstantLearningRateScheduler(learningRate),
            maxGradientNorm);
    }

    /// <summary>
    /// Starts a session training a copy of the one-neuron network.
    /// </summary>
    /// <param name="optimizer">
    /// The optimizer to train with.
    /// </param>
    /// <param name="learningRate">
    /// The constant learning rate to train at.
    /// </param>
    /// <param name="costFunction">
    /// The cost function to train against, or <see langword="null"/> for mean squared error.
    /// </param>
    /// <returns>
    /// Returns the session.
    /// </returns>
    public static TrainingSession StartSession(
        IOptimizer optimizer,
        float learningRate = LearningRate,
        ICostFunction? costFunction = null)
    {
        return TrainingSession.Start(
            BuildNeuronNetwork(),
            CreateSettings(optimizer, learningRate, costFunction));
    }

    /// <summary>
    /// Reads the one layer of a network built by <see cref="BuildNeuronNetwork"/>.
    /// </summary>
    /// <param name="network">
    /// The network to read.
    /// </param>
    /// <returns>
    /// Returns the layer's document.
    /// </returns>
    public static NeuralLayerDocument ReadLayer(INeuralNetwork network) =>
        NetworkDocument.FromNetwork(network).NeuralLayers.Single();

    /// <summary>
    /// Reads every neural layer of a network, in the order the network holds them.
    /// </summary>
    /// <param name="network">
    /// The network to read.
    /// </param>
    /// <returns>
    /// Returns the layers' documents, for a network of more than the one layer <see cref="ReadLayer"/> serves.
    /// </returns>
    public static IReadOnlyList<NeuralLayerDocument> ReadLayers(INeuralNetwork network) =>
        NetworkDocument.FromNetwork(network).NeuralLayers;

    /// <summary>
    /// Asserts that two networks of the same shape hold the same weights and biases.
    /// </summary>
    /// <param name="network">
    /// The network whose values are expected.
    /// </param>
    /// <param name="other">
    /// The network to check.
    /// </param>
    public static void AssertNetworksAgree(INeuralNetwork network, INeuralNetwork other)
    {
        IReadOnlyList<NeuralLayerDocument> layers = NetworkDocument.FromNetwork(network).NeuralLayers;
        IReadOnlyList<NeuralLayerDocument> otherLayers = NetworkDocument.FromNetwork(other).NeuralLayers;

        Assert.AreEqual(layers.Count, otherLayers.Count, "Count of layers.");

        for (int layerIndex = 0; layerIndex < layers.Count; layerIndex++)
        {
            for (int rowIndex = 0; rowIndex < layers[layerIndex].WeightMatrix.Count; rowIndex++)
            {
                NetworkAssert.ValuesAre(
                    otherLayers[layerIndex].WeightMatrix[rowIndex],
                    layers[layerIndex].WeightMatrix[rowIndex].Select(weight => (double)weight),
                    $"weight in row {rowIndex} of layer {layerIndex} of");
            }

            NetworkAssert.ValuesAre(
                otherLayers[layerIndex].BiasVector,
                layers[layerIndex].BiasVector.Select(bias => (double)bias),
                $"bias of layer {layerIndex} of");
        }
    }

    #endregion
}
