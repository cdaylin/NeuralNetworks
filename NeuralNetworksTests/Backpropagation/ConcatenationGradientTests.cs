using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.Backpropagation.Optimization;
using Daylin.NeuralNetworks.Serialization;
using Daylin.NeuralNetworks.WeightInitialization;

namespace Daylin.NeuralNetworks.Backpropagation;

/// <summary>
/// Tests the gradients a concatenation layer hands back to the layer before it.
/// </summary>
/// <remarks>
/// A concatenation layer gathers the gradients of the layers inside it and passes them on as one vector, in the
/// order those layers are held.  Every other test of a concatenation puts it first in its network, where there
/// is no preceding layer and this half of the layer never runs -- while ReversiApp's convolutional agent puts
/// one after a routing layer, so the untested half was the one in use.
/// </remarks>
[TestClass]
public sealed class ConcatenationGradientTests
{
    #region Public

    [TestMethod]
    public async Task AConcatenationPassesEachNestedLayersGradientsToThePrecedingLayerInOrderAsync()
    {
        INeuralNetwork network = BuildNetwork();

        TrainingSession session = TrainingSession.Adopt(
            network,
            TrainingFixture.CreateSettings(Optimizers.GradientDescent, LearningRate));

        await session.TrainSampleAsync(Input, Targets);

        // The hidden layer's two neurons are fed to the two nested layers, in order:  neuron 0 to the first
        //  nested layer, neuron 1 to the second.  So each hidden neuron's gradient comes from its own nested
        //  layer, and a concatenation that mixed them up would move both hidden neurons by the other's share.
        (IReadOnlyList<double> expectedHiddenWeights, IReadOnlyList<double> expectedHiddenBiases) =
            CalculateExpectedHiddenLayer();

        NeuralLayerDocument hiddenLayer = NetworkDocument.FromNetwork(network).NeuralLayers[0];

        NetworkAssert.ValuesAre(hiddenLayer.WeightMatrix[0], expectedHiddenWeights.Take(2), "weight of neuron 0 of");
        NetworkAssert.ValuesAre(hiddenLayer.WeightMatrix[1], expectedHiddenWeights.Skip(2), "weight of neuron 1 of");
        NetworkAssert.ValuesAre(hiddenLayer.BiasVector, expectedHiddenBiases, "bias of");
    }

    [TestMethod]
    public async Task AConcatenationOfLayersOfDifferentWidthsSplitsTheGradientsWhereTheyJoinAsync()
    {
        // Both nested layers being one value wide leaves the splitting of the gradient vector untested:  an
        //  offset or width error looks the same as no error.  Here the first nested layer takes two of the three
        //  values the preceding layer produces and the second takes one, and the expected weights are computed
        //  from the rules the library states rather than read back from it.
        INeuralNetwork network = BuildUnevenNetwork();

        TrainingSession session = TrainingSession.Adopt(
            network,
            TrainingFixture.CreateSettings(Optimizers.GradientDescent, LearningRate));

        await session.TrainSampleAsync(Input, Targets);

        (IReadOnlyList<double> expectedHiddenWeights, IReadOnlyList<double> expectedHiddenBiases) =
            CalculateExpectedUnevenHiddenLayer();

        IReadOnlyList<NeuralLayerDocument> layers = NetworkDocument.FromNetwork(network).NeuralLayers;

        // The hidden layer is what pins the split:  its first two neurons take their gradients from the first
        //  nested layer and its third from the second, so a vector split at the wrong offset -- or either
        //  layer's slice handed back in the wrong order -- moves a hidden neuron by another's share.  The three
        //  hidden neurons are deliberately unlike one another, so no two shares are interchangeable.
        for (int neuronIndex = 0; neuronIndex < 3; neuronIndex++)
        {
            NetworkAssert.ValuesAre(
                layers[0].WeightMatrix[neuronIndex],
                expectedHiddenWeights.Skip(neuronIndex * 2).Take(2),
                $"weight of hidden neuron {neuronIndex} of");
        }

        NetworkAssert.ValuesAre(layers[0].BiasVector, expectedHiddenBiases, "bias of the hidden layer of");
    }

    #endregion

    #region Private Static

    private static float LearningRate => TrainingFixture.LearningRate;

    private static IReadOnlyList<float> Input { get; } = [0.5f, -0.25f];

    private static IReadOnlyList<float> Targets { get; } = [0.75f, -0.5f];

    /// <summary>
    /// The hidden layer's weights, two for each of its two neurons.
    /// </summary>
    private static IReadOnlyList<double> HiddenWeights { get; } = [0.4, -0.3, 0.2, 0.6];

    /// <summary>
    /// The hidden layer's biases, one for each of its two neurons.
    /// </summary>
    private static IReadOnlyList<double> HiddenBiases { get; } = [0.1, -0.2];

    /// <summary>
    /// The weight each nested output layer holds, one connection each.
    /// </summary>
    private static IReadOnlyList<double> OutputWeights { get; } = [0.7, -0.45];

    /// <summary>
    /// The bias each nested output layer holds.
    /// </summary>
    private static IReadOnlyList<double> OutputBiases { get; } = [-0.05, 0.15];

    /// <summary>
    /// Builds a network whose concatenation layer has a layer before it:  two inputs, a dense layer of two
    /// neurons, then a concatenation of two dense layers of one neuron each.
    /// </summary>
    /// <returns>
    /// Returns the network, with every weight and bias written in.
    /// </returns>
    private static INeuralNetwork BuildNetwork()
    {
        IConnectionWeightInitializer weightInitializer = new GlorotUniformWeightInitializer(seed: 11);

        NeuralNetworkFactory factory = new();

        INeuralNetwork network = factory.CreateNetwork(
            factory.CreateDenseLayer(2, 2, ActivationFunctions.Tanh, weightInitializer),
            factory.Concatenate(
                factory.CreateDenseLayer(1, 1, ActivationFunctions.Tanh, weightInitializer),
                factory.CreateDenseLayer(1, 1, ActivationFunctions.Tanh, weightInitializer)));

        NetworkDocument document = NetworkDocument.FromNetwork(network);

        document.NeuralLayers[0].SetWeightMatrix(
            HiddenWeights.Take(2).Select(weight => (float)weight),
            HiddenWeights.Skip(2).Select(weight => (float)weight));

        document.NeuralLayers[0].SetBiasVector(HiddenBiases.Select(bias => (float)bias));

        for (int layerIndex = 0; layerIndex < 2; layerIndex++)
        {
            document.NeuralLayers[layerIndex + 1].SetWeightMatrix([(float)OutputWeights[layerIndex]]);
            document.NeuralLayers[layerIndex + 1].SetBiasVector((float)OutputBiases[layerIndex]);
        }

        return document.BuildNetwork();
    }

    /// <summary>
    /// The hidden layer's weights for the uneven network, two for each of its three neurons.
    /// </summary>
    /// <remarks>
    /// Deliberately unlike one another, so that a hidden neuron moved by another neuron's share of the gradient
    /// lands somewhere the expected values do not.
    /// </remarks>
    private static IReadOnlyList<double> UnevenHiddenWeights { get; } = [0.4, -0.3, 0.2, 0.6, -0.5, 0.35];

    /// <summary>
    /// The hidden layer's biases for the uneven network, one for each of its three neurons.
    /// </summary>
    private static IReadOnlyList<double> UnevenHiddenBiases { get; } = [0.1, -0.2, 0.25];

    /// <summary>
    /// The first nested layer's two weights, over the first two hidden neurons.
    /// </summary>
    private static IReadOnlyList<double> UnevenFirstNestedWeights { get; } = [0.7, -0.45];

    /// <summary>
    /// The second nested layer's one weight, over the third hidden neuron.
    /// </summary>
    private static double UnevenSecondNestedWeight => 0.55;

    /// <summary>
    /// The bias each nested layer of the uneven network holds.
    /// </summary>
    private static IReadOnlyList<double> UnevenNestedBiases { get; } = [-0.05, 0.15];

    /// <summary>
    /// Builds a network whose concatenation holds layers of different widths:  two inputs, a dense layer of
    /// three neurons, then a concatenation of a dense layer over two of them and one over the third.
    /// </summary>
    /// <returns>
    /// Returns the network, with every weight and bias written in.
    /// </returns>
    private static INeuralNetwork BuildUnevenNetwork()
    {
        IConnectionWeightInitializer weightInitializer = new GlorotUniformWeightInitializer(seed: 17);

        NeuralNetworkFactory factory = new();

        INeuralNetwork network = factory.CreateNetwork(
            factory.CreateDenseLayer(2, 3, ActivationFunctions.Tanh, weightInitializer),
            factory.Concatenate(
                factory.CreateDenseLayer(2, 1, ActivationFunctions.Tanh, weightInitializer),
                factory.CreateDenseLayer(1, 1, ActivationFunctions.Tanh, weightInitializer)));

        NetworkDocument document = NetworkDocument.FromNetwork(network);

        document.NeuralLayers[0].SetWeightMatrix(
            UnevenHiddenWeights.Take(2).Select(weight => (float)weight),
            UnevenHiddenWeights.Skip(2).Take(2).Select(weight => (float)weight),
            UnevenHiddenWeights.Skip(4).Select(weight => (float)weight));

        document.NeuralLayers[0].SetBiasVector(UnevenHiddenBiases.Select(bias => (float)bias));

        document.NeuralLayers[1].SetWeightMatrix(UnevenFirstNestedWeights.Select(weight => (float)weight));
        document.NeuralLayers[1].SetBiasVector((float)UnevenNestedBiases[0]);

        document.NeuralLayers[2].SetWeightMatrix([(float)UnevenSecondNestedWeight]);
        document.NeuralLayers[2].SetBiasVector((float)UnevenNestedBiases[1]);

        return document.BuildNetwork();
    }

    /// <summary>
    /// Calculates where one training step leaves the uneven network's hidden layer, in double precision.
    /// </summary>
    /// <remarks>
    /// The first nested layer takes hidden neurons 0 and 1 and the second takes neuron 2, so each hidden
    /// neuron's gradient arrives through the weight its own nested layer holds for it.
    /// </remarks>
    /// <returns>
    /// Returns the hidden layer's weights, two for each of its three neurons, and its biases.
    /// </returns>
    private static (IReadOnlyList<double> Weights, IReadOnlyList<double> Biases) CalculateExpectedUnevenHiddenLayer()
    {
        double[] hidden = new double[3];

        for (int neuronIndex = 0; neuronIndex < 3; neuronIndex++)
        {
            hidden[neuronIndex] = Math.Tanh(
                (UnevenHiddenWeights[(neuronIndex * 2) + 0] * Input[0])
                + (UnevenHiddenWeights[(neuronIndex * 2) + 1] * Input[1])
                + UnevenHiddenBiases[neuronIndex]);
        }

        double firstOutput = Math.Tanh(
            (UnevenFirstNestedWeights[0] * hidden[0])
            + (UnevenFirstNestedWeights[1] * hidden[1])
            + UnevenNestedBiases[0]);

        double secondOutput = Math.Tanh((UnevenSecondNestedWeight * hidden[2]) + UnevenNestedBiases[1]);

        double firstOutputGradient = (firstOutput - Targets[0]) * (1 - (firstOutput * firstOutput));
        double secondOutputGradient = (secondOutput - Targets[1]) * (1 - (secondOutput * secondOutput));

        // What the concatenation hands back for each hidden neuron, in the hidden layer's own order.
        double[] hiddenGradients =
        [
            firstOutputGradient * UnevenFirstNestedWeights[0] * (1 - (hidden[0] * hidden[0])),
            firstOutputGradient * UnevenFirstNestedWeights[1] * (1 - (hidden[1] * hidden[1])),
            secondOutputGradient * UnevenSecondNestedWeight * (1 - (hidden[2] * hidden[2])),
        ];

        double[] weights = [.. UnevenHiddenWeights];
        double[] biases = [.. UnevenHiddenBiases];

        for (int neuronIndex = 0; neuronIndex < 3; neuronIndex++)
        {
            weights[(neuronIndex * 2) + 0] -= LearningRate * hiddenGradients[neuronIndex] * Input[0];
            weights[(neuronIndex * 2) + 1] -= LearningRate * hiddenGradients[neuronIndex] * Input[1];

            biases[neuronIndex] -= LearningRate * hiddenGradients[neuronIndex];
        }

        return (weights, biases);
    }

    /// <summary>
    /// Calculates where one training step leaves the hidden layer, in double precision.
    /// </summary>
    /// <remarks>
    /// Written out from the rules the library states rather than from the library's own arithmetic:  each
    /// output value's gradient reaches the hidden neuron it came from, through the nested layer's one weight.
    /// </remarks>
    /// <returns>
    /// Returns the hidden layer's weights, two for each neuron, and its biases.
    /// </returns>
    private static (IReadOnlyList<double> Weights, IReadOnlyList<double> Biases) CalculateExpectedHiddenLayer()
    {
        double[] weights = [.. HiddenWeights];
        double[] biases = [.. HiddenBiases];

        for (int neuronIndex = 0; neuronIndex < 2; neuronIndex++)
        {
            double hidden = Math.Tanh(
                (HiddenWeights[(neuronIndex * 2) + 0] * Input[0])
                + (HiddenWeights[(neuronIndex * 2) + 1] * Input[1])
                + HiddenBiases[neuronIndex]);

            double output = Math.Tanh((OutputWeights[neuronIndex] * hidden) + OutputBiases[neuronIndex]);

            double outputGradient = (output - Targets[neuronIndex]) * (1 - (output * output));

            // What the concatenation hands back for this hidden neuron.
            double hiddenGradient = outputGradient * OutputWeights[neuronIndex] * (1 - (hidden * hidden));

            weights[(neuronIndex * 2) + 0] -= LearningRate * hiddenGradient * Input[0];
            weights[(neuronIndex * 2) + 1] -= LearningRate * hiddenGradient * Input[1];

            biases[neuronIndex] -= LearningRate * hiddenGradient;
        }

        return (weights, biases);
    }

    #endregion
}
