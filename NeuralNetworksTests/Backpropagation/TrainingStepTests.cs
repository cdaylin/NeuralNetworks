using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.Backpropagation.Cost;
using Daylin.NeuralNetworks.Backpropagation.LearningRate;
using Daylin.NeuralNetworks.Backpropagation.Optimization;
using Daylin.NeuralNetworks.LayerTemplates;
using Daylin.NeuralNetworks.Serialization;
using Daylin.NeuralNetworks.WeightInitialization;

namespace Daylin.NeuralNetworks.Backpropagation;

/// <summary>
/// Tests the weights and biases a single training step leaves behind.
/// </summary>
/// <remarks>
/// Where <see cref="BackpropagationTests"/> asks whether a network can be trained at all, these tests ask what
/// one step of training does.  Each network is given known weights through its serialized document, trained on
/// one sample, and read back the same way.
/// <para>
/// The expected values are computed here from the rules the library states:  a cost derivative returns the
/// partial derivative of the cost function, which for the mean squared error is <c>actual - target</c>; a
/// neuron's gradient is that value multiplied by its activation derivative; a weight and a bias each move
/// <em>against</em> the gradient, by the learning rate multiplied by it, the weight also by the input value
/// it carries; and the gradient reaching a preceding layer is the sum over the neurons it feeds of their
/// gradients multiplied by the connection weights <em>as they stood before the update</em>.
/// </para>
/// </remarks>
[TestClass]
public class TrainingStepTests
{
    #region Private Static

    private static float LearningRate => 0.5f;

    /// <summary>
    /// The derivative of the hyperbolic tangent, at the weighted input whose activation is given.
    /// </summary>
    /// <param name="activation">
    /// The activation calculated for the weighted input.
    /// </param>
    /// <returns>
    /// Returns the derivative of the hyperbolic tangent at that weighted input.
    /// </returns>
    private static double TanhDerivative(double activation) => 1 - (activation * activation);

    private static void AssertWeightsAndBiases(
        NeuralLayerDocument layer,
        IReadOnlyList<IReadOnlyList<double>> expectedWeightMatrix,
        IReadOnlyList<double> expectedBiasVector)
    {
        Assert.AreEqual(expectedWeightMatrix.Count, layer.WeightMatrix.Count, "Count of weight rows.");

        for (int rowIndex = 0; rowIndex < expectedWeightMatrix.Count; rowIndex++)
        {
            NetworkAssert.ValuesAre(
                layer.WeightMatrix[rowIndex],
                expectedWeightMatrix[rowIndex],
                $"weight in row {rowIndex} of");
        }

        NetworkAssert.ValuesAre(layer.BiasVector, expectedBiasVector, "bias of");
    }

    #endregion

    #region Public

    [TestMethod]
    public async Task EachWeightMovesByTheLearningRateTheGradientAndTheInputItCarriesAsync()
    {
        NeuralNetworkBuilder builder = new(inputCount: 2);

        builder.AddDenseLayer(1, ActivationFunctions.Tanh, WeightInitializer);

        INeuralNetwork network = BuildNetwork(builder, layer =>
        {
            layer.SetWeightMatrix([0.5f, -0.25f]);
            layer.SetBiasVector(0.1f);
        });

        await TrainSampleAsync(network, [0.4f, -0.6f], [1f]);

        double activation = Math.Tanh((0.5 * 0.4) + (-0.25 * -0.6) + 0.1);
        double gradient = (activation - 1.0) * TanhDerivative(activation);

        AssertWeightsAndBiases(
            ReadLayer(network, 0),
            [[0.5 - (0.5 * gradient * 0.4), -0.25 - (0.5 * gradient * -0.6)]],
            [0.1 - (0.5 * gradient)]);
    }

    [TestMethod]
    public async Task APrecedingLayerReceivesGradientsThroughTheWeightsHeldBeforeTheUpdateAsync()
    {
        NeuralNetworkBuilder builder = new(inputCount: 2);

        builder.AddDenseLayer(2, ActivationFunctions.Tanh, WeightInitializer);
        builder.AddDenseLayer(1, ActivationFunctions.Tanh, WeightInitializer);

        NetworkDocument document = NetworkDocument.FromNetwork(builder.BuildNetwork());

        document.NeuralLayers[0].SetWeightMatrix([0.5f, -0.25f], [1f, 0.75f]);
        document.NeuralLayers[0].SetBiasVector(0.1f, -0.2f);
        document.NeuralLayers[1].SetWeightMatrix([0.6f, -0.4f]);
        document.NeuralLayers[1].SetBiasVector(0.05f);

        INeuralNetwork network = document.BuildNetwork();

        await TrainSampleAsync(network, [0.4f, -0.6f], [1f]);

        double firstHiddenValue = Math.Tanh((0.5 * 0.4) + (-0.25 * -0.6) + 0.1);
        double secondHiddenValue = Math.Tanh((1.0 * 0.4) + (0.75 * -0.6) - 0.2);

        double outputValue = Math.Tanh((0.6 * firstHiddenValue) + (-0.4 * secondHiddenValue) + 0.05);
        double outputGradient = (outputValue - 1.0) * TanhDerivative(outputValue);

        // The gradients reaching the hidden neurons are carried by the output layer's weights as they stood
        // before this step moved them.
        double firstHiddenGradient = outputGradient * 0.6 * TanhDerivative(firstHiddenValue);
        double secondHiddenGradient = outputGradient * -0.4 * TanhDerivative(secondHiddenValue);

        AssertWeightsAndBiases(
            ReadLayer(network, 1),
            [
                [
                    0.6 - (0.5 * outputGradient * firstHiddenValue),
                    -0.4 - (0.5 * outputGradient * secondHiddenValue),
                ],
            ],
            [0.05 - (0.5 * outputGradient)]);

        AssertWeightsAndBiases(
            ReadLayer(network, 0),
            [
                [0.5 - (0.5 * firstHiddenGradient * 0.4), -0.25 - (0.5 * firstHiddenGradient * -0.6)],
                [1.0 - (0.5 * secondHiddenGradient * 0.4), 0.75 - (0.5 * secondHiddenGradient * -0.6)],
            ],
            [0.1 - (0.5 * firstHiddenGradient), -0.2 - (0.5 * secondHiddenGradient)]);
    }

    [TestMethod]
    public async Task ConvolutionalLayerGathersOneUpdatePerSpanIntoItsSharedWeightsAsync()
    {
        NeuralNetworkBuilder builder = new(inputCount: 4);

        builder.AddConvolutionalLayer(2, ActivationFunctions.Tanh, WeightInitializer);

        INeuralNetwork network = BuildNetwork(builder, layer =>
        {
            layer.SetWeightMatrix([0.5f, -0.25f]);
            layer.SetBiasVector(0.1f);
        });

        await TrainSampleAsync(network, [0.4f, -0.6f, 0.8f, 1f], [1f, -1f]);

        double firstValue = Math.Tanh((0.5 * 0.4) + (-0.25 * -0.6) + 0.1);
        double secondValue = Math.Tanh((0.5 * 0.8) + (-0.25 * 1.0) + 0.1);

        double firstGradient = (firstValue - 1.0) * TanhDerivative(firstValue);
        double secondGradient = (secondValue - -1.0) * TanhDerivative(secondValue);

        // Both output values share the one bias and the one pair of weights, so each shared value takes the
        // update owed by every span.
        AssertWeightsAndBiases(
            ReadLayer(network, 0),
            [
                [
                    0.5 - (0.5 * ((firstGradient * 0.4) + (secondGradient * 0.8))),
                    -0.25 - (0.5 * ((firstGradient * -0.6) + (secondGradient * 1.0))),
                ],
            ],
            [0.1 - (0.5 * (firstGradient + secondGradient))]);
    }

    [TestMethod]
    public async Task SparseLayerUpdatesOnlyTheConnectionsItHasAsync()
    {
        ConnectionMaskBuilder maskBuilder = new(inputCount: 3, outputCount: 2);

        maskBuilder.SetIsConnectionMasked(neuronIndex: 0, connectionIndex: 1, isMasked: true);

        NeuralNetworkBuilder builder = new(inputCount: 3);

        builder.AddSparseLayer(maskBuilder.Build(), ActivationFunctions.Tanh, WeightInitializer);

        INeuralNetwork network = BuildNetwork(builder, layer =>
        {
            layer.SetWeightMatrix([0.5f, -0.25f], [1f, 0.75f, 0.5f]);
            layer.SetBiasVector(0.1f, -0.2f);
        });

        await TrainSampleAsync(network, [0.4f, -0.6f, 0.8f], [1f, -1f]);

        double firstValue = Math.Tanh((0.5 * 0.4) + (-0.25 * 0.8) + 0.1);
        double secondValue = Math.Tanh((1.0 * 0.4) + (0.75 * -0.6) + (0.5 * 0.8) - 0.2);

        double firstGradient = (firstValue - 1.0) * TanhDerivative(firstValue);
        double secondGradient = (secondValue - -1.0) * TanhDerivative(secondValue);

        // The first neuron holds two weights, carrying the first and third input values:  the second input
        // value is masked from it, and moves neither of them.
        AssertWeightsAndBiases(
            ReadLayer(network, 0),
            [
                [0.5 - (0.5 * firstGradient * 0.4), -0.25 - (0.5 * firstGradient * 0.8)],
                [
                    1.0 - (0.5 * secondGradient * 0.4),
                    0.75 - (0.5 * secondGradient * -0.6),
                    0.5 - (0.5 * secondGradient * 0.8),
                ],
            ],
            [0.1 - (0.5 * firstGradient), -0.2 - (0.5 * secondGradient)]);
    }

    [TestMethod]
    public async Task ConnectionRoutingLayerReturnsEachGradientToTheNeuronItCameFromAsync()
    {
        NeuralNetworkBuilder builder = new(inputCount: 2);

        builder.AddDenseLayer(3, ActivationFunctions.Tanh, WeightInitializer);

        // A rotation rather than a swap, so that reading the map the wrong way round moves the gradients
        // somewhere other than where they came from.
        builder.AddConnectionRoutingLayer(new IndexMap([2, 0, 1]));

        INeuralNetwork network = BuildNetwork(builder, layer =>
        {
            layer.SetWeightMatrix([0.5f, -0.25f], [1f, 0.75f], [-0.5f, 0.25f]);
            layer.SetBiasVector(0.1f, -0.2f, 0.3f);
        });

        await TrainSampleAsync(network, [0.4f, -0.6f], [1f, -1f, 0.5f]);

        double firstValue = Math.Tanh((0.5 * 0.4) + (-0.25 * -0.6) + 0.1);
        double secondValue = Math.Tanh((1.0 * 0.4) + (0.75 * -0.6) - 0.2);
        double thirdValue = Math.Tanh((-0.5 * 0.4) + (0.25 * -0.6) + 0.3);

        // Each neuron's value leaves the network at the index the map holds for it, so it is the target at
        // that index the neuron answers for:  the first neuron's value leaves third, the second's first, and
        // the third's second.
        double firstGradient = (firstValue - 0.5) * TanhDerivative(firstValue);
        double secondGradient = (secondValue - 1.0) * TanhDerivative(secondValue);
        double thirdGradient = (thirdValue - -1.0) * TanhDerivative(thirdValue);

        AssertWeightsAndBiases(
            ReadLayer(network, 0),
            [
                [0.5 - (0.5 * firstGradient * 0.4), -0.25 - (0.5 * firstGradient * -0.6)],
                [1.0 - (0.5 * secondGradient * 0.4), 0.75 - (0.5 * secondGradient * -0.6)],
                [-0.5 - (0.5 * thirdGradient * 0.4), 0.25 - (0.5 * thirdGradient * -0.6)],
            ],
            [
                0.1 - (0.5 * firstGradient),
                -0.2 - (0.5 * secondGradient),
                0.3 - (0.5 * thirdGradient),
            ]);
    }

    [TestMethod]
    public async Task ConcatenationLayerGivesEachNestedLayerOnlyItsOwnGradientsAsync()
    {
        NeuralNetworkFactory factory = new();

        INetworkLayerTemplate layers = factory.Concatenate(
            factory.CreateDenseLayer(2, 1, ActivationFunctions.Tanh, WeightInitializer),
            factory.CreateDenseLayer(2, 1, ActivationFunctions.Tanh, WeightInitializer));

        NetworkDocument document = NetworkDocument.FromNetwork(factory.CreateNetwork(layers));

        document.NeuralLayers[0].SetWeightMatrix([0.5f, -0.25f]);
        document.NeuralLayers[0].SetBiasVector(0.1f);
        document.NeuralLayers[1].SetWeightMatrix([1f, 0.75f]);
        document.NeuralLayers[1].SetBiasVector(-0.2f);

        INeuralNetwork network = document.BuildNetwork();

        await TrainSampleAsync(network, [0.4f, -0.6f, 0.4f, -0.6f], [1f, -1f]);

        double firstValue = Math.Tanh((0.5 * 0.4) + (-0.25 * -0.6) + 0.1);
        double secondValue = Math.Tanh((1.0 * 0.4) + (0.75 * -0.6) - 0.2);

        double firstGradient = (firstValue - 1.0) * TanhDerivative(firstValue);
        double secondGradient = (secondValue - -1.0) * TanhDerivative(secondValue);

        // Both nested layers receive the same input values, so only their own gradients distinguish them.
        AssertWeightsAndBiases(
            ReadLayer(network, 0),
            [[0.5 - (0.5 * firstGradient * 0.4), -0.25 - (0.5 * firstGradient * -0.6)]],
            [0.1 - (0.5 * firstGradient)]);

        AssertWeightsAndBiases(
            ReadLayer(network, 1),
            [[1.0 - (0.5 * secondGradient * 0.4), 0.75 - (0.5 * secondGradient * -0.6)]],
            [-0.2 - (0.5 * secondGradient)]);
    }

    [TestMethod]
    public async Task TrainingRejectsInputAndTargetCountsTheNetworkDoesNotHaveAsync()
    {
        NeuralNetworkBuilder builder = new(inputCount: 2);

        builder.AddDenseLayer(1, ActivationFunctions.Tanh, WeightInitializer);

        INeuralNetwork network = builder.BuildNetwork();

        await Assert.ThrowsExactlyAsync<ArgumentOutOfRangeException>(
            () => TrainSampleAsync(network, [0.4f], [1f]));

        await Assert.ThrowsExactlyAsync<ArgumentOutOfRangeException>(
            () => TrainSampleAsync(network, [0.4f, -0.6f], [1f, 1f]));
    }

    #endregion

    #region Private

    private IConnectionWeightInitializer WeightInitializer { get; } = new GlorotUniformWeightInitializer(seed: 1);

    private TrainingSettings Settings { get; } = new(
        CostFunctions.MeanSquaredError,
        Optimizers.GradientDescent,
        LearningRateSchedulers.ConstantLearningRateScheduler(LearningRate));

    /// <summary>
    /// Trains a network on one sample, in place, so that the caller reads the network it passed.
    /// </summary>
    /// <param name="network">
    /// The network to train.
    /// </param>
    /// <param name="input">
    /// Input values.
    /// </param>
    /// <param name="targetOutput">
    /// Desired output values.
    /// </param>
    /// <returns>
    /// Returns a task representing this <c>async</c> operation.
    /// </returns>
    private async Task TrainSampleAsync(
        INeuralNetwork network,
        IReadOnlyList<float> input,
        IReadOnlyList<float> targetOutput)
    {
        await TrainingSession.Adopt(network, Settings).TrainSampleAsync(input, targetOutput);
    }

    /// <summary>
    /// Builds a network of one neural layer and gives that layer known weights and biases.
    /// </summary>
    /// <param name="builder">
    /// A builder holding the layer to build.
    /// </param>
    /// <param name="configureLayer">
    /// An action that sets the layer's weights and biases.
    /// </param>
    /// <returns>
    /// Returns the network built from the configured document.
    /// </returns>
    private INeuralNetwork BuildNetwork(NeuralNetworkBuilder builder, Action<NeuralLayerDocument> configureLayer)
    {
        NetworkDocument document = NetworkDocument.FromNetwork(builder.BuildNetwork());

        configureLayer(document.NeuralLayers.Single());

        return document.BuildNetwork();
    }

    private NeuralLayerDocument ReadLayer(INeuralNetwork network, int layerIndex)
    {
        return NetworkDocument.FromNetwork(network).NeuralLayers[layerIndex];
    }

    #endregion
}
