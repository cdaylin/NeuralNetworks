using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.WeightInitialization;
using Daylin.Utilities.Exceptions;

namespace Daylin.NeuralNetworks.Backpropagation;

/// <summary>
/// Tests for training neural networks via backpropagation.
/// </summary>
/// <remarks>
/// The purpose of these tests is to verify that the backpropagation training algorithm "works", rather than to
/// test specific aspects of the training algorithm.
/// <para>
/// These tests are in a sense performance tests, in that they verify that simple neural networks can be
/// successfully trained to perform basic algorithms within a reasonable number of training epochs.
/// </para><para>
/// Backpropagation training can be considered to be a form of gradient descent search algorithm.  Gradient
/// descent is not guaranteed to find an optimal solution, due to the potential to "get stuck" in local
/// optima.  Whether a particular network escapes a local optimum depends on its initial connection weights,
/// so each network here is built from a seeded weight initializer.  Every run therefore trains exactly the
/// same networks, and a failure is a real signal rather than an unlucky draw.
/// </para><para>
/// The seeds are arbitrary, and are meant to stay that way.  A seed chosen because it passes would prove
/// only that it still passes, and would leave a regression that broke most other seeds undetected.  So when
/// a seed fails, the network configuration is what should change -- as it did for ReLU, where an arbitrary
/// seed revealed that the hidden layer was too small.
/// </para>
/// </remarks>
[TestClass]
public class BackpropagationTests
{
    public TestContext? TestContext { get; set; }

    [TestMethod]
    [DataRow("EANAF", 101)]
    [DataRow("ReLU", 102)]
    [DataRow("Tanh", 103)]
    [DataRow("Softsign", 104)]
    [DataRow("Sigmoid", 105)]
    public async Task NetworkLearnsIdentityGateAsync(string activationFunctionName, int seed)
    {
        await TestWithSingleHiddenLayerAsync(
            testData: BackpropagationTestData.Identity,
            neuronCount: 4,
            activationFunctionName: activationFunctionName,
            seed: seed);
    }

    [TestMethod]
    [DataRow("EANAF", 201)]
    [DataRow("ReLU", 202)]
    [DataRow("Tanh", 203)]
    [DataRow("Softsign", 204)]
    [DataRow("Sigmoid", 205)]
    public async Task NetworkLearnsNotGateAsync(string activationFunctionName, int seed)
    {
        await TestWithSingleHiddenLayerAsync(
            testData: BackpropagationTestData.Not,
            neuronCount: 4,
            activationFunctionName: activationFunctionName,
            seed: seed);
    }

    [TestMethod]
    [DataRow("EANAF", 301)]
    [DataRow("ReLU", 302)]
    [DataRow("Tanh", 303)]
    [DataRow("Softsign", 304)]
    [DataRow("Sigmoid", 305)]
    public async Task NetworkLearnsAndGateAsync(string activationFunctionName, int seed)
    {
        await TestWithSingleHiddenLayerAsync(
            testData: BackpropagationTestData.And,
            neuronCount: 4,
            activationFunctionName: activationFunctionName,
            seed: seed);
    }

    [TestMethod]
    [DataRow("EANAF", 401)]
    [DataRow("ReLU", 402)]
    [DataRow("Tanh", 403)]
    [DataRow("Softsign", 404)]
    [DataRow("Sigmoid", 405)]
    public async Task NetworkLearnsExclusiveOrGateAsync(string activationFunctionName, int seed)
    {
        await TestWithSingleHiddenLayerAsync(
            testData: BackpropagationTestData.XOr,
            neuronCount: 8,
            activationFunctionName: activationFunctionName,
            seed: seed);
    }

    [TestMethod]
    public async Task DeepNetworkOfTanhLayersLearnsExclusiveOrAsync()
    {
        BackpropagationTestData data = BackpropagationTestData.XOr;

        await TestAsync(
            testData: data,
            seed: 501,
            (4, ActivationFunctions.Tanh),
            (4, ActivationFunctions.Tanh),
            (data.OutputCount, ActivationFunctions.Tanh));
    }

    [TestMethod]
    public async Task DeepNetworkOfMixedActivationFunctionsLearnsExclusiveOrAsync()
    {
        BackpropagationTestData data = BackpropagationTestData.XOr;

        // The ReLU layer is sized for the same reason as minReluNeuronCount below, measured against this
        // architecture rather than that one: over 60 seeds, 4 neurons converged 80% of the time, 6 and 8
        // both 98%, and 10 or more always.  A Softsign layer ahead of ReLU makes it considerably more
        // reliable than it is on the raw inputs, where 8 neurons converged only 88% of the time.
        await TestAsync(
            testData: data,
            seed: 502,
            (4, ActivationFunctions.Softsign),
            (10, ActivationFunctions.Relu),
            (data.OutputCount, ActivationFunctions.Tanh));
    }

    [TestMethod]
    public async Task DeepNetworkOfNarrowEanafLayersLearnsExclusiveOrAsync()
    {
        BackpropagationTestData data = BackpropagationTestData.XOr;

        await TestAsync(
            testData: data,
            seed: 503,
            (2, ActivationFunctions.Eanaf),
            (4, ActivationFunctions.Eanaf),
            (data.OutputCount, ActivationFunctions.Eanaf));
    }

    [TestMethod]
    public async Task DeepNetworkOfFourHiddenLayersLearnsExclusiveOrAsync()
    {
        BackpropagationTestData data = BackpropagationTestData.XOr;

        await TestAsync(
            testData: data,
            seed: 504,
            (4, ActivationFunctions.Tanh),
            (4, ActivationFunctions.Eanaf),
            (4, ActivationFunctions.Tanh),
            (4, ActivationFunctions.Eanaf),
            (data.OutputCount, ActivationFunctions.Tanh));
    }

    private async Task TestWithSingleHiddenLayerAsync(
        BackpropagationTestData testData,
        int neuronCount,
        string activationFunctionName,
        int seed)
    {
        IActivationFunction activationFunction = GetActivationFunction(activationFunctionName);

        // Use Tanh for output layer.  (ReLU does not work well.  Tanh tends to train quickly.)
        (int Count, IActivationFunction Tanh) outputLayer = (testData.OutputCount, ActivationFunctions.Tanh);

        // ReLU performs poorly with small numbers of neurons in a layer (due to 'dying' neurons?).  Measured
        // on XOr over 60 seeds: 4 neurons converged 48% of the time, 6 77%, 8 88%, 10 97%, and 12 or more
        // always.  Twelve also trains fastest of those, at a median of 158 epochs against 178 at eight.
        int minReluNeuronCount = 12;

        int hiddenNeuronCount = activationFunction.ShortName == "ReLU"
            ? Math.Max(neuronCount, minReluNeuronCount)
            : neuronCount;

        await TestAsync(testData, seed, (hiddenNeuronCount, activationFunction), outputLayer);
    }

    private async Task TestAsync(
        BackpropagationTestData testData,
        int seed,
        params IEnumerable<(int neuronCount, IActivationFunction activationFunction)> args)
    {
        NeuralNetworkBuilder networkBuilder = new(testData.InputCount);

        // Each layer is initialized from its own generator, as it is when no seed is supplied.  Deriving
        // those seeds from one source keeps the layers independent while making the whole network
        // reproducible from a single number.
        Random seedSource = new(seed);

        foreach ((int outputCount, IActivationFunction activationFunction) in args)
        {
            networkBuilder.AddDenseLayer(
                outputCount: outputCount,
                activationFunction: activationFunction,
                weightInitializer: GetWeightInitializer(activationFunction, seedSource.Next()));
        }

        INeuralNetwork neuralNetwork = networkBuilder.BuildNetwork();

        await TestAsync(
            neuralNetwork,
            testData,
            networkDescription: $"{testData.ShortName}:  {neuralNetwork.ShortDescription}");
    }

    private async Task TestAsync(
        INeuralNetwork neuralNetwork,
        BackpropagationTestData testData,
        BackpropagationSettings? settings = null,
        int maxEpochCount = 50000,
        string? networkDescription = null)
    {
        TestContext!.WriteLine(null);
        TestContext!.WriteLine(
            $"Starting to train neural network {(networkDescription is null ? "" : $": {networkDescription}")}.");

        settings ??= new BackpropagationSettings(
            learningRate: .1f,
            errorFunction: ErrorFunctions.MeanSquaredError);

        int epochCount = 0;

        while (!await IsOutputWithinToleranceAsync(neuralNetwork, testData.InputData, testData.TargetData))
        {
            Assert.IsTrue(
                epochCount < maxEpochCount,
                $"The neural network was not successfully trained after {maxEpochCount} epochs.");

            epochCount++;

            for (int index = 0; index < testData.InputData.Count; index++)
            {
                await neuralNetwork.TrainAsync(
                    settings,
                    testData.InputData[index],
                    testData.TargetData[index]);
            }
        }

        TestContext.WriteLine($"The neural network was successfully trained after {epochCount} epochs.");
    }

    private IActivationFunction GetActivationFunction(string shortName)
    {
        return shortName switch
        {
            "EANAF" => ActivationFunctions.Eanaf,
            "ReLU" => ActivationFunctions.Relu,
            "Tanh" => ActivationFunctions.Tanh,
            "Softsign" => ActivationFunctions.Softsign,
            "Sigmoid" => ActivationFunctions.Sigmoid,
            _ => throw UnhandledCaseException.Create(shortName),
        };
    }

    private IConnectionWeightInitializer GetWeightInitializer(IActivationFunction activationFunction, int seed)
    {
        // The asymmetric activation functions pair with He initialization, the symmetric ones with Glorot.
        return activationFunction.ShortName switch
        {
            "EANAF" or "ReLU" => new HeUniformWeightInitializer(seed: seed),
            "Tanh" or "Softsign" or "Sigmoid" => new GlorotUniformWeightInitializer(seed: seed),
            _ => throw UnhandledCaseException.Create(activationFunction.ShortName),
        };
    }

    private async Task<bool> IsOutputWithinToleranceAsync(
        INeuralNetwork neuralNetwork,
        IReadOnlyList<IReadOnlyList<float>> input,
        IReadOnlyList<IReadOnlyList<float>> target,
        float equalityTolerance = 0.1f)
    {
        for (int index = 0; index < input.Count; index++)
        {
            IEnumerable<float> output = await neuralNetwork.CalculateOutputAsync(input[index]);

            Assert.IsFalse(output.Any(float.IsNaN), "The neural network output 'NaN'.");

            if (!AreApproximatelyEqual(output, target[index], equalityTolerance))
                return false;
        }

        return true;
    }

    private bool AreApproximatelyEqual(IEnumerable<float> first, IEnumerable<float> second, float tolerance)
    {
        return first
            .Zip(second)
            .All(pair => Math.Abs(pair.First - pair.Second) <= tolerance);
    }
}
