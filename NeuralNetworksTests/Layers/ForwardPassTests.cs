using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.LayerTemplates;
using Daylin.NeuralNetworks.Serialization;
using Daylin.NeuralNetworks.WeightInitialization;

namespace Daylin.NeuralNetworks.Layers;

/// <summary>
/// Tests that each type of network layer calculates the output its specification describes.
/// </summary>
/// <remarks>
/// Each network is given known weights and biases through its serialized document, and the output expected of
/// it is computed from the arithmetic the layer is specified to perform, in double precision.
/// </remarks>
[TestClass]
public class ForwardPassTests
{
    #region Public

    [TestMethod]
    public async Task DenseLayerAppliesWeightsBiasAndActivationAsync()
    {
        NeuralNetworkBuilder builder = new(inputCount: 2);

        builder.AddDenseLayer(2, ActivationFunctions.Tanh, WeightInitializer);

        NetworkDocument document = NetworkDocument.FromNetwork(builder.BuildNetwork());

        document.NeuralLayers[0].SetWeightMatrix([0.5f, -0.25f], [1f, 0.75f]);
        document.NeuralLayers[0].SetBiasVector(0.1f, -0.2f);

        await NetworkAssert.OutputIsAsync(
            document.BuildNetwork(),
            [0.4f, -0.6f],
            Math.Tanh((0.5 * 0.4) + (-0.25 * -0.6) + 0.1),
            Math.Tanh((1.0 * 0.4) + (0.75 * -0.6) - 0.2));
    }

    [TestMethod]
    public async Task ConvolutionalLayerAppliesOneSetOfWeightsToEachSpanAsync()
    {
        NeuralNetworkBuilder builder = new(inputCount: 4);

        builder.AddConvolutionalLayer(2, ActivationFunctions.Tanh, WeightInitializer);

        NetworkDocument document = NetworkDocument.FromNetwork(builder.BuildNetwork());

        // The two output values share one bias and one pair of weights, each covering two consecutive inputs.
        document.NeuralLayers[0].SetWeightMatrix([0.5f, -0.25f]);
        document.NeuralLayers[0].SetBiasVector(0.1f);

        await NetworkAssert.OutputIsAsync(
            document.BuildNetwork(),
            [0.4f, -0.6f, 0.8f, 1f],
            Math.Tanh((0.5 * 0.4) + (-0.25 * -0.6) + 0.1),
            Math.Tanh((0.5 * 0.8) + (-0.25 * 1.0) + 0.1));
    }

    [TestMethod]
    public async Task SparseLayerCarriesNothingThroughAMaskedConnectionAsync()
    {
        INeuralNetwork network = BuildSparseNetwork();

        // The first neuron's two weights carry the first and third input values, the second being masked.
        double firstOutput = Math.Tanh((0.5 * 0.4) + (-0.25 * 0.8) + 0.1);

        await NetworkAssert.OutputIsAsync(
            network,
            [0.4f, -0.6f, 0.8f],
            firstOutput,
            Math.Tanh((1.0 * 0.4) + (0.75 * -0.6) + (0.5 * 0.8) - 0.2));

        // The second input is masked from the first neuron, so changing it cannot change the first output.
        await NetworkAssert.OutputIsAsync(
            network,
            [0.4f, -1f, 0.8f],
            firstOutput,
            Math.Tanh((1.0 * 0.4) + (0.75 * -1.0) + (0.5 * 0.8) - 0.2));
    }

    [TestMethod]
    public async Task ConnectionRoutingLayerMovesEachValueToItsMappedIndexAsync()
    {
        NeuralNetworkFactory factory = new();

        // The value at each position moves to the index the map holds at that position.
        INeuralNetwork network = factory.CreateNetwork(
            factory.CreateConnectionRoutingLayer(new IndexMap([2, 0, 1])));

        await NetworkAssert.OutputIsAsync(network, [0.1f, 0.2f, 0.3f], 0.2, 0.3, 0.1);
    }

    [TestMethod]
    public async Task PassthroughLayerForwardsEveryValueUnchangedAsync()
    {
        NeuralNetworkFactory factory = new();

        INeuralNetwork network = factory.CreateNetwork(factory.CreatePassthroughLayer(3));

        await NetworkAssert.OutputIsAsync(network, [0.1f, -0.2f, 0.3f], 0.1, -0.2, 0.3);
    }

    [TestMethod]
    public async Task ConcatenationLayerGivesEachNestedLayerItsOwnShareOfTheInputAsync()
    {
        NeuralNetworkFactory factory = new();

        INetworkLayerTemplate layers = factory.Concatenate(
            factory.CreateDenseLayer(2, 1, ActivationFunctions.Tanh, WeightInitializer),
            factory.CreateDenseLayer(3, 2, ActivationFunctions.Tanh, WeightInitializer));

        NetworkDocument document = NetworkDocument.FromNetwork(factory.CreateNetwork(layers));

        document.NeuralLayers[0].SetWeightMatrix([0.5f, -0.25f]);
        document.NeuralLayers[0].SetBiasVector(0.1f);
        document.NeuralLayers[1].SetWeightMatrix([1f, 0.75f, -0.5f], [0.25f, 0.25f, 0.25f]);
        document.NeuralLayers[1].SetBiasVector(-0.2f, 0f);

        // The first two inputs reach the first nested layer, and the last three reach the second.
        await NetworkAssert.OutputIsAsync(
            document.BuildNetwork(),
            [0.4f, -0.6f, 0.8f, 1f, -1f],
            Math.Tanh((0.5 * 0.4) + (-0.25 * -0.6) + 0.1),
            Math.Tanh((1.0 * 0.8) + (0.75 * 1.0) + (-0.5 * -1.0) - 0.2),
            Math.Tanh((0.25 * 0.8) + (0.25 * 1.0) + (0.25 * -1.0)));
    }

    [TestMethod]
    public async Task SequenceLayerFeedsEachLayersOutputToTheNextAsync()
    {
        NeuralNetworkBuilder builder = new(inputCount: 2);

        builder.AddDenseLayer(2, ActivationFunctions.Tanh, WeightInitializer);
        builder.AddDenseLayer(1, ActivationFunctions.Tanh, WeightInitializer);

        NetworkDocument document = NetworkDocument.FromNetwork(builder.BuildNetwork());

        document.NeuralLayers[0].SetWeightMatrix([0.5f, -0.25f], [1f, 0.75f]);
        document.NeuralLayers[0].SetBiasVector(0.1f, -0.2f);
        document.NeuralLayers[1].SetWeightMatrix([0.6f, -0.4f]);
        document.NeuralLayers[1].SetBiasVector(0.05f);

        double firstHiddenValue = Math.Tanh((0.5 * 0.4) + (-0.25 * -0.6) + 0.1);
        double secondHiddenValue = Math.Tanh((1.0 * 0.4) + (0.75 * -0.6) - 0.2);

        await NetworkAssert.OutputIsAsync(
            document.BuildNetwork(),
            [0.4f, -0.6f],
            Math.Tanh((0.6 * firstHiddenValue) + (-0.4 * secondHiddenValue) + 0.05));
    }

    #endregion

    #region Private

    private IConnectionWeightInitializer WeightInitializer { get; } = new GlorotUniformWeightInitializer(seed: 1);

    /// <summary>
    /// Builds a network of one sparse layer whose first neuron is masked from the third input value.
    /// </summary>
    /// <returns>
    /// Returns a network of three input values and two output values, with known weights and biases.
    /// </returns>
    private INeuralNetwork BuildSparseNetwork()
    {
        ConnectionMaskBuilder maskBuilder = new(inputCount: 3, outputCount: 2);

        maskBuilder.SetIsConnectionMasked(neuronIndex: 0, connectionIndex: 1, isMasked: true);

        NeuralNetworkBuilder builder = new(inputCount: 3);

        builder.AddSparseLayer(maskBuilder.Build(), ActivationFunctions.Tanh, WeightInitializer);

        NetworkDocument document = NetworkDocument.FromNetwork(builder.BuildNetwork());

        // The first neuron holds two weights rather than three, the masked connection having none.
        document.NeuralLayers[0].SetWeightMatrix([0.5f, -0.25f], [1f, 0.75f, 0.5f]);
        document.NeuralLayers[0].SetBiasVector(0.1f, -0.2f);

        return document.BuildNetwork();
    }

    #endregion
}
