using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.Backpropagation.LearningRate;
using Daylin.NeuralNetworks.Backpropagation.Optimization;
using Daylin.NeuralNetworks.Serialization;
using Daylin.NeuralNetworks.WeightInitialization;
using Daylin.TestUtilities;

namespace Daylin.NeuralNetworks.Backpropagation;

/// <summary>
/// Tests the optimizers that move a network's parameters.
/// </summary>
/// <remarks>
/// Each test trains a network of one neuron over several batches and compares it against the same arithmetic
/// carried out here in double precision, from the formulas the optimizers document rather than from the
/// library's own way of reaching them.
/// </remarks>
[TestClass]
public class OptimizerTests
{
    #region Private Static

    private static float LearningRate => 0.5f;

    private static int StepCount => 3;

    private static IReadOnlyList<IReadOnlyList<float>> BatchInput { get; } = [[0.4f, -0.6f], [-0.2f, 0.8f]];

    private static IReadOnlyList<IReadOnlyList<float>> BatchTargets { get; } = [[1f], [-1f]];

    private static IReadOnlyList<IReadOnlyList<double>> ReferenceInput { get; } = [[0.4, -0.6], [-0.2, 0.8]];

    private static IReadOnlyList<double> ReferenceTargets { get; } = [1, -1];

    #endregion

    #region Public

    [TestMethod]
    public async Task MomentumAddsEachBatchsGradientToTheVelocityThatSurvivedTheLastAsync()
    {
        float momentum = 0.5f;

        TrainingSession session = await TrainAsync(Optimizers.Momentum(momentum));

        ReferenceNeuron neuron = CreateReferenceNeuron();

        double[] weightVelocities = new double[neuron.Weights.Length];
        double biasVelocity = 0;

        for (int stepIndex = 0; stepIndex < StepCount; stepIndex++)
        {
            (IReadOnlyList<double> weightGradients, double biasGradient) =
                neuron.CalculateBatchGradients(ReferenceInput, ReferenceTargets);

            for (int index = 0; index < neuron.Weights.Length; index++)
            {
                weightVelocities[index] = (momentum * weightVelocities[index]) + weightGradients[index];

                neuron.Weights[index] -= LearningRate * weightVelocities[index];
            }

            biasVelocity = (momentum * biasVelocity) + biasGradient;

            neuron.Bias -= LearningRate * biasVelocity;
        }

        AssertNetworkMatches(session.Network, neuron);
    }

    [TestMethod]
    public async Task MomentumOfZeroMovesAParameterAsPlainGradientDescentDoesAsync()
    {
        TrainingSession withMomentum = await TrainAsync(Optimizers.Momentum(momentum: 0));
        TrainingSession withoutMomentum = await TrainAsync(Optimizers.GradientDescent());

        AssertNetworksAgree(withMomentum.Network, withoutMomentum.Network);
    }

    [TestMethod]
    public async Task AdamDividesEachCorrectedMeanGradientByItsCorrectedMagnitudeAsync()
    {
        float firstMomentDecay = 0.9f;
        float secondMomentDecay = 0.999f;
        float epsilon = 1e-8f;

        TrainingSession session = await TrainAsync(
            Optimizers.Adam(firstMomentDecay, secondMomentDecay, epsilon));

        ReferenceNeuron neuron = CreateReferenceNeuron();

        double[] firstMoments = new double[neuron.Weights.Length + 1];
        double[] secondMoments = new double[neuron.Weights.Length + 1];

        for (int stepIndex = 0; stepIndex < StepCount; stepIndex++)
        {
            (IReadOnlyList<double> weightGradients, double biasGradient) =
                neuron.CalculateBatchGradients(ReferenceInput, ReferenceTargets);

            double firstMomentRemainder = 1 - Math.Pow(firstMomentDecay, stepIndex + 1);
            double secondMomentRemainder = 1 - Math.Pow(secondMomentDecay, stepIndex + 1);

            for (int index = 0; index < neuron.Weights.Length; index++)
            {
                neuron.Weights[index] -= CalculateAdamStep(
                    firstMoments,
                    secondMoments,
                    index,
                    weightGradients[index],
                    (firstMomentDecay, secondMomentDecay, epsilon),
                    (firstMomentRemainder, secondMomentRemainder));
            }

            neuron.Bias -= CalculateAdamStep(
                firstMoments,
                secondMoments,
                neuron.Weights.Length,
                biasGradient,
                (firstMomentDecay, secondMomentDecay, epsilon),
                (firstMomentRemainder, secondMomentRemainder));
        }

        AssertNetworkMatches(session.Network, neuron);
    }

    [TestMethod]
    public async Task AdamsFirstStepIsTheWholeLearningRateHoweverLargeTheGradientAsync()
    {
        TrainingSession session = await TrainAsync(Optimizers.Adam(), stepCount: 1);

        ReferenceNeuron neuron = CreateReferenceNeuron();

        NeuralLayerDocument layer = ReadLayer(session.Network);

        // Both corrected moments come from the one gradient, so their ratio is one and every parameter moves
        // the whole learning rate, against the direction its gradient pointed.
        (IReadOnlyList<double> weightGradients, double biasGradient) =
            neuron.CalculateBatchGradients(ReferenceInput, ReferenceTargets);

        NetworkAssert.ValuesAre(
            layer.WeightMatrix.Single(),
            neuron.Weights.Zip(weightGradients, (weight, gradient) => weight - (LearningRate * Math.Sign(gradient))),
            "weight of");

        NetworkAssert.ValuesAre(
            layer.BiasVector,
            [neuron.Bias - (LearningRate * Math.Sign(biasGradient))],
            "bias of");
    }

    [TestMethod]
    public async Task ASessionStartedFromATrainedNetworkCarriesNoneOfItsOptimizersStateAsync()
    {
        TrainingSession trainedByAdam = await TrainAsync(Optimizers.Adam(), stepCount: 2);

        TrainingSession continuedWithMomentum = StartSession(trainedByAdam.Network, Optimizers.Momentum());
        TrainingSession continuedWithDescent = StartSession(trainedByAdam.Network, Optimizers.GradientDescent());

        await continuedWithMomentum.TrainBatchAsync(BatchInput, BatchTargets);
        await continuedWithDescent.TrainBatchAsync(BatchInput, BatchTargets);

        // Momentum carrying no velocity yet moves a parameter exactly as plain gradient descent does, which it
        // would not if the new session had taken over what Adam left behind.  An optimizer belongs to its
        // session, so training under another one means a new session, and a new session begins with nothing.
        AssertNetworksAgree(continuedWithMomentum.Network, continuedWithDescent.Network);
    }

    [TestMethod]
    public void AnOptimizerRejectsHyperparametersOutsideItsRange()
    {
        Asserter.AssertThat(() => Optimizers.Momentum(momentum: 1)).Throws<ArgumentOutOfRangeException>();
        Asserter.AssertThat(() => Optimizers.Momentum(momentum: -0.1f)).Throws<ArgumentOutOfRangeException>();
        Asserter.AssertThat(() => Optimizers.Adam(firstMomentDecay: 1)).Throws<ArgumentOutOfRangeException>();
        Asserter.AssertThat(() => Optimizers.Adam(secondMomentDecay: 1)).Throws<ArgumentOutOfRangeException>();
        Asserter.AssertThat(() => Optimizers.Adam(epsilon: 0)).Throws<ArgumentOutOfRangeException>();
    }

    #endregion

    #region Private

    private IConnectionWeightInitializer WeightInitializer { get; } = new GlorotUniformWeightInitializer(seed: 1);

    /// <summary>
    /// Calculates the distance a parameter moves under Adam, advancing its moment estimates.
    /// </summary>
    /// <param name="firstMoments">
    /// The mean gradient carried for each parameter.
    /// </param>
    /// <param name="secondMoments">
    /// The mean squared gradient carried for each parameter.
    /// </param>
    /// <param name="index">
    /// Index of the parameter within those arrays.
    /// </param>
    /// <param name="gradient">
    /// The gradient the batch calculated for the parameter.
    /// </param>
    /// <param name="hyperparameters">
    /// The optimizer's decay rates and epsilon.
    /// </param>
    /// <param name="remainders">
    /// The share of each moment that the steps so far have not decayed away.
    /// </param>
    /// <returns>
    /// Returns the distance the parameter moves against its gradient.
    /// </returns>
    private double CalculateAdamStep(
        double[] firstMoments,
        double[] secondMoments,
        int index,
        double gradient,
        (double FirstMomentDecay, double SecondMomentDecay, double Epsilon) hyperparameters,
        (double FirstMoment, double SecondMoment) remainders)
    {
        firstMoments[index] = (hyperparameters.FirstMomentDecay * firstMoments[index])
            + ((1 - hyperparameters.FirstMomentDecay) * gradient);

        secondMoments[index] = (hyperparameters.SecondMomentDecay * secondMoments[index])
            + ((1 - hyperparameters.SecondMomentDecay) * gradient * gradient);

        double correctedFirstMoment = firstMoments[index] / remainders.FirstMoment;
        double correctedSecondMoment = secondMoments[index] / remainders.SecondMoment;

        return LearningRate * correctedFirstMoment / (Math.Sqrt(correctedSecondMoment) + hyperparameters.Epsilon);
    }

    private ReferenceNeuron CreateReferenceNeuron() => new([0.5, -0.25], 0.1);

    /// <summary>
    /// Builds a network of one neuron over two input values, with the weights and bias
    /// <see cref="CreateReferenceNeuron"/> describes.
    /// </summary>
    /// <returns>
    /// Returns the network.
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

    /// <summary>
    /// Trains a fresh session over several batches of the same samples.
    /// </summary>
    /// <param name="optimizer">
    /// The optimizer to train with.
    /// </param>
    /// <param name="stepCount">
    /// Number of batches to train, or <see langword="null"/> for <see cref="StepCount"/>.
    /// </param>
    /// <returns>
    /// Returns the session, whose network carries the result.
    /// </returns>
    private async Task<TrainingSession> TrainAsync(IOptimizer optimizer, int? stepCount = null)
    {
        TrainingSession session = StartSession(BuildNetwork(), optimizer);

        for (int stepIndex = 0; stepIndex < (stepCount ?? StepCount); stepIndex++)
            await session.TrainBatchAsync(BatchInput, BatchTargets);

        return session;
    }

    private TrainingSession StartSession(INeuralNetwork network, IOptimizer optimizer)
    {
        return TrainingSession.Start(
            network,
            new TrainingSettings(
                CostDerivatives.MeanSquaredError,
                optimizer,
                LearningRateSchedulers.ConstantLearningRateScheduler(LearningRate)));
    }

    private NeuralLayerDocument ReadLayer(INeuralNetwork network) =>
        NetworkDocument.FromNetwork(network).NeuralLayers.Single();

    private void AssertNetworkMatches(INeuralNetwork network, ReferenceNeuron neuron)
    {
        NeuralLayerDocument layer = ReadLayer(network);

        NetworkAssert.ValuesAre(layer.WeightMatrix.Single(), neuron.Weights, "weight of");
        NetworkAssert.ValuesAre(layer.BiasVector, [neuron.Bias], "bias of");
    }

    private void AssertNetworksAgree(INeuralNetwork network, INeuralNetwork other)
    {
        NeuralLayerDocument layer = ReadLayer(network);

        NetworkAssert.ValuesAre(
            ReadLayer(other).WeightMatrix.Single(),
            layer.WeightMatrix.Single().Select(weight => (double)weight),
            "weight of");

        NetworkAssert.ValuesAre(
            ReadLayer(other).BiasVector,
            layer.BiasVector.Select(bias => (double)bias),
            "bias of");
    }

    #endregion
}
