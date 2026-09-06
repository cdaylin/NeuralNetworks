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
/// optima.  Due to the randomization of initial connection weights in each neural network, these tests
/// are nondeterministic.  Therefore, it is possible that a test may occassionally fail due to the training
/// algorithm failing to escape a local optimum.  A very occassional test failure due to a the maximum number
/// of epochs being exceeded does not necessarily indicate a defect.
/// </para>
/// </remarks>
[TestClass]
[TestCategory("Nondeterministic")]
public class BackpropagationTests
{
    public TestContext? TestContext { get; set; }

    [TestMethod]
    public async Task TestIdentityGateAsync()
    {
        await TestWithSingleHiddenLayerAsync(
            testData: BackpropagationTestData.Identity,
            neuronCount: 4);
    }

    [TestMethod]
    public async Task TestNotGateAsync()
    {
        await TestWithSingleHiddenLayerAsync(
            testData: BackpropagationTestData.Not,
            neuronCount: 4);
    }

    [TestMethod]
    public async Task TestAndGateAsync()
    {
        await TestWithSingleHiddenLayerAsync(
            testData: BackpropagationTestData.And,
            neuronCount: 4);
    }

    [TestMethod]
    public async Task TestXOrGateAsync()
    {
        await TestWithSingleHiddenLayerAsync(
            testData: BackpropagationTestData.XOr,
            neuronCount: 8);
    }

    [TestMethod]
    public async Task TestDeepNetworksAsync()
    {
        BackpropagationTestData data = BackpropagationTestData.XOr;

        await TestAsync(
            testData: data,
            (4, ActivationFunctions.Tanh),
            (4, ActivationFunctions.Tanh),
            (data.OutputCount, ActivationFunctions.Tanh));

        // The ReLU layer is sized for the same reason as minReluNeuronCount above, measured against this
        // architecture rather than that one: over 60 seeded runs, 4 neurons converged 80% of the time, 6 and
        // 8 both 98%, and 10 or more always.  A Softsign layer ahead of ReLU makes it considerably more
        // reliable than it is on the raw inputs, where 8 neurons converged only 88% of the time.
        await TestAsync(
            testData: data,
            (4, ActivationFunctions.Softsign),
            (10, ActivationFunctions.Relu),
            (data.OutputCount, ActivationFunctions.Tanh));

        await TestAsync(
            testData: data,
            (2, ActivationFunctions.Eanaf),
            (4, ActivationFunctions.Eanaf),
            (data.OutputCount, ActivationFunctions.Eanaf));

        await TestAsync(
            testData: data,
            (4, ActivationFunctions.Tanh),
            (4, ActivationFunctions.Eanaf),
            (4, ActivationFunctions.Tanh),
            (4, ActivationFunctions.Eanaf),
            (data.OutputCount, ActivationFunctions.Tanh));
    }

    private async Task TestWithSingleHiddenLayerAsync(BackpropagationTestData testData, int neuronCount)
    {
        // Use Tanh for output layer.  (ReLU does not work well.  Tanh tends to train quickly.)
        (int Count, IActivationFunction Tanh) outputLayer = (testData.OutputCount, ActivationFunctions.Tanh);

        // ReLU performs poorly with small numbers of neurons in a layer (due to 'dying' neurons?).  Measured
        // on XOr over 60 seeded runs: 4 neurons converged 48% of the time, 6 77%, 8 88%, 10 97%, and 12 or
        // more always.  Twelve also trains fastest of those, at a median of 158 epochs against 178 at eight.
        int minReluNeuronCount = 12;

        // asymmetric activation functions
        await TestAsync(testData, (neuronCount, ActivationFunctions.Eanaf), outputLayer);
        await TestAsync(testData, (Math.Max(neuronCount, minReluNeuronCount), ActivationFunctions.Relu), outputLayer);

        // symmetric activation functions
        await TestAsync(testData, (neuronCount, ActivationFunctions.Tanh), outputLayer);
        await TestAsync(testData, (neuronCount, ActivationFunctions.Softsign), outputLayer);
        await TestAsync(testData, (neuronCount, ActivationFunctions.Sigmoid), outputLayer);
    }

    private async Task TestAsync(
        BackpropagationTestData testData,
        params IEnumerable<(int neuronCount, IActivationFunction activationFunction)> args)
    {
        NeuralNetworkBuilder networkBuilder = new(testData.InputCount);

        foreach ((int outputCount, IActivationFunction activationFunction) in args)
        {
            networkBuilder.AddDenseLayer(
                outputCount: outputCount,
                activationFunction: activationFunction,
                weightInitializer: GetWeightInitializer(activationFunction));
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

    private IConnectionWeightInitializer GetWeightInitializer(IActivationFunction activationFunction)
    {
        return activationFunction.ShortName switch
        {
            "EANAF" or "ReLU" => WeightInitializers.He,
            "Tanh" or "Softsign" or "Sigmoid" => WeightInitializers.Glorot,
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
