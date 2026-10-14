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

    private static float LearningRate => TrainingFixture.LearningRate;

    private static int StepCount => 3;

    private static IReadOnlyList<IReadOnlyList<float>> BatchInput => TrainingFixture.BatchInput;

    private static IReadOnlyList<IReadOnlyList<float>> BatchTargets => TrainingFixture.BatchTargets;

    private static IReadOnlyList<IReadOnlyList<double>> ReferenceInput =>
        TrainingFixture.BatchInputInDoublePrecision;

    private static IReadOnlyList<double> ReferenceTargets => TrainingFixture.BatchTargetsInDoublePrecision;

    #endregion

    #region Public

    [TestMethod]
    public async Task MomentumAddsTheGradientOfEachBatchToTheVelocityThatSurvivedTheLastAsync()
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
        TrainingSession withoutMomentum = await TrainAsync(Optimizers.GradientDescent);

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
    public async Task AdamAddsItsEpsilonToTheRootRatherThanUnderItAsync()
    {
        // With the published epsilon of 1e-8 the two placements differ by about a ten-millionth of a step,
        // which no tolerance here can see, so this trains at an epsilon large enough to tell them apart and
        // asserts that the two forms really do differ before asserting which one the library takes.
        float epsilon = 0.05f;
        float firstMomentDecay = 0.9f;
        float secondMomentDecay = 0.999f;

        TrainingSession session = await TrainAsync(
            Optimizers.Adam(firstMomentDecay, secondMomentDecay, epsilon),
            stepCount: 1);

        ReferenceNeuron neuron = CreateReferenceNeuron();

        (IReadOnlyList<double> weightGradients, double biasGradient) =
            neuron.CalculateBatchGradients(ReferenceInput, ReferenceTargets);

        double firstMomentRemainder = 1 - firstMomentDecay;
        double secondMomentRemainder = 1 - secondMomentDecay;

        double correctedFirstMoment = (1 - firstMomentDecay) * biasGradient / firstMomentRemainder;
        double correctedSecondMoment =
            (1 - secondMomentDecay) * biasGradient * biasGradient / secondMomentRemainder;

        double stepWithEpsilonOutsideTheRoot =
            LearningRate * correctedFirstMoment / (Math.Sqrt(correctedSecondMoment) + epsilon);

        double stepWithEpsilonUnderTheRoot =
            LearningRate * correctedFirstMoment / Math.Sqrt(correctedSecondMoment + epsilon);

        Assert.IsTrue(
            Math.Abs(stepWithEpsilonOutsideTheRoot - stepWithEpsilonUnderTheRoot) > NetworkAssert.Tolerance * 10,
            "The two placements of epsilon are too close together for this test to tell them apart.");

        NeuralLayerDocument layer = ReadLayer(session.Network);

        NetworkAssert.ValuesAre(
            layer.BiasVector,
            [neuron.Bias - stepWithEpsilonOutsideTheRoot],
            "bias of");

        for (int index = 0; index < neuron.Weights.Length; index++)
        {
            double correctedFirstWeightMoment =
                (1 - firstMomentDecay) * weightGradients[index] / firstMomentRemainder;

            double correctedSecondWeightMoment =
                (1 - secondMomentDecay) * weightGradients[index] * weightGradients[index] / secondMomentRemainder;

            NetworkAssert.ValuesAre(
                [layer.WeightMatrix.Single()[index]],
                [neuron.Weights[index]
                    - (LearningRate * correctedFirstWeightMoment
                        / (Math.Sqrt(correctedSecondWeightMoment) + epsilon))],
                $"weight {index} of");
        }
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
        TrainingSession continuedWithDescent = StartSession(trainedByAdam.Network, Optimizers.GradientDescent);

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

    [TestMethod]
    public void AnOptimizerRefusesStateAnotherOptimizerCreated()
    {
        // State of another kind must be refused by name rather than dereferenced:  a caller holding it has made
        //  a mistake it cannot see from a NullReferenceException raised inside the step.
        float[] parameters = [0.5f, -0.25f];
        float[] gradients = [0.1f, 0.1f];

        IOptimizerState? momentumState = Optimizers.Momentum().CreateState(parameters.Length);
        IOptimizerState? adamState = Optimizers.Adam().CreateState(parameters.Length);

        Assert.ThrowsExactly<ArgumentException>(
            () => Optimizers.Adam().ApplyGradients(momentumState, parameters, gradients, learningRate: 0.5f),
            "Adam accepted momentum's state.");

        Assert.ThrowsExactly<ArgumentException>(
            () => Optimizers.Momentum().ApplyGradients(adamState, parameters, gradients, learningRate: 0.5f),
            "Momentum accepted Adam's state.");

        Assert.ThrowsExactly<ArgumentException>(
            () => Optimizers.Momentum().ApplyGradients(state: null, parameters, gradients, learningRate: 0.5f),
            "Momentum accepted no state at all.");

        // State of its own kind, sized for another group:  the branch a same-kind mismatch reaches.
        Assert.ThrowsExactly<ArgumentException>(
            () => Optimizers.Adam().ApplyGradients(
                Optimizers.Adam().CreateState(parameters.Length + 1),
                parameters,
                gradients,
                learningRate: 0.5f),
            "Adam accepted state sized for another group.");

        // And the span's own length check.
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(
            () => Optimizers.Adam().ApplyGradients(adamState, parameters, [0.1f], learningRate: 0.5f),
            "Adam accepted a gradient for every other parameter.");
    }

    [TestMethod]
    public void GradientDescentRefusesStateRatherThanIgnoringIt()
    {
        // Gradient descent keeps nothing, so state handed to it is a caller's mistake:  ignoring it would leave
        //  that caller believing a velocity was being carried.  Unreachable through a session, since a group's
        //  state is settled against the optimizer before the step, so a test is the only thing that reaches it.
        float[] parameters = [0.5f, -0.25f];
        float[] gradients = [0.1f, 0.1f];

        IOptimizerState? momentumState = Optimizers.Momentum().CreateState(parameters.Length);

        ArgumentException refusal = Assert.ThrowsExactly<ArgumentException>(
            () => Optimizers.GradientDescent.ApplyGradients(
                momentumState,
                parameters,
                gradients,
                learningRate: 0.5f),
            "Gradient descent accepted momentum's state.");

        StringAssert.Contains(refusal.Message, "carries no state", "The refusal says why it cannot be used.");

        // Its own state is null, which it must still accept, and the step is the plain one.
        Optimizers.GradientDescent.ApplyGradients(
            Optimizers.GradientDescent.CreateState(parameters.Length),
            parameters,
            gradients,
            learningRate: 0.5f);

        NetworkAssert.ValuesAre([parameters[0], parameters[1]], [0.45, -0.3], "parameter");
    }

    [TestMethod]
    public void AnOptimizerOfOneKindAcceptsAnothersStateWhereTheyDifferOnlyInHyperparameters()
    {
        // Recorded rather than fixed:  a state carries its kind and its length and nothing else, so this is a
        //  warm restart under new hyperparameters rather than an error.  The test exists because the interface
        //  said "state it created itself" until 2026-10-13, which claimed more than the code checks.
        IOptimizerState? adamState = Optimizers.Adam(epsilon: 1e-8f).CreateState(parameterCount: 2);

        Assert.IsTrue(
            Optimizers.Adam(epsilon: 1e-3f).CanApplyTo(adamState, parameterCount: 2),
            "Adam refused the state of an Adam with another epsilon.");

        Assert.IsFalse(
            Optimizers.Momentum().CanApplyTo(adamState, parameterCount: 2),
            "Momentum accepted Adam's state.");

        Assert.IsFalse(
            Optimizers.Adam().CanApplyTo(adamState, parameterCount: 3),
            "Adam accepted state sized for another group.");
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

    private ReferenceNeuron CreateReferenceNeuron() => TrainingFixture.CreateReferenceNeuron();

    /// <summary>
    /// Builds a network of one neuron over two input values, with the weights and bias
    /// <see cref="CreateReferenceNeuron"/> describes.
    /// </summary>
    /// <returns>
    /// Returns the network.
    /// </returns>
    private INeuralNetwork BuildNetwork() => TrainingFixture.BuildNeuronNetwork();

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

    private TrainingSession StartSession(INeuralNetwork network, IOptimizer optimizer) =>
        TrainingSession.Start(network, TrainingFixture.CreateSettings(optimizer, LearningRate));

    private NeuralLayerDocument ReadLayer(INeuralNetwork network) => TrainingFixture.ReadLayer(network);

    private void AssertNetworkMatches(INeuralNetwork network, ReferenceNeuron neuron)
    {
        NeuralLayerDocument layer = ReadLayer(network);

        NetworkAssert.ValuesAre(layer.WeightMatrix.Single(), neuron.Weights, "weight of");
        NetworkAssert.ValuesAre(layer.BiasVector, [neuron.Bias], "bias of");
    }

    private void AssertNetworksAgree(INeuralNetwork network, INeuralNetwork other) =>
        TrainingFixture.AssertNetworksAgree(network, other);

    #endregion
}
