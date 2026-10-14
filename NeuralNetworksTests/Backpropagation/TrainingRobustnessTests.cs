using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.Backpropagation.Cost;
using Daylin.NeuralNetworks.Backpropagation.LearningRate;
using Daylin.NeuralNetworks.Backpropagation.Optimization;
using Daylin.NeuralNetworks.Serialization;
using Daylin.NeuralNetworks.WeightInitialization;

using System.Text.Json.Nodes;

namespace Daylin.NeuralNetworks.Backpropagation;

/// <summary>
/// Tests what training does when something goes wrong, and what it leaves behind afterwards.
/// </summary>
/// <remarks>
/// Each of these fails quietly if it fails at all:  a network left holding a failed batch's gradients, a limit
/// on the gradients that does not apply, a network that cannot be trained, a cost figure that does not describe
/// what was learned from, and a session that will not train again after a batch that moved nothing.  Written as
/// the class's question rather than as a list of its contents, which goes stale as tests are added.
/// </remarks>
[TestClass]
public sealed class TrainingRobustnessTests
{
    #region Public

    [TestMethod]
    public async Task ABatchWhoseStepFailsBeforeMovingAnythingLeavesTheNetworkAsItFoundItAsync()
    {
        // A step that fails on the network's first parameter group has moved nothing:  the network must be
        //  exactly as it was, and the batch's gradients must not reach the next batch.
        TrainingSession session = TrainingFixture.StartSession(ThrowingOptimizer.OnFirstGroup());

        NeuralLayerDocument before = TrainingFixture.ReadLayer(session.Network);

        await Assert.ThrowsExactlyAsync<InvalidOperationException>(
            async () => await session.TrainSampleAsync(TrainingFixture.Input, TrainingFixture.Target));

        NetworkAssert.ValuesAre(
            TrainingFixture.ReadLayer(session.Network).WeightMatrix.Single(),
            before.WeightMatrix.Single().Select(weight => (double)weight),
            "weight after the failed batch of");

        NetworkAssert.ValuesAre(
            TrainingFixture.ReadLayer(session.Network).BiasVector,
            before.BiasVector.Select(bias => (double)bias),
            "bias after the failed batch of");

        // Trained now by an optimizer that works, the network must move exactly as it would from a batch of its
        //  own:  the failed batch's gradients are gone rather than added to the next.
        TrainingSession afterwards = TrainingSession.Adopt(
            session.Network,
            TrainingFixture.CreateSettings(Optimizers.GradientDescent));

        await afterwards.TrainSampleAsync(TrainingFixture.Input, TrainingFixture.Target);

        TrainingSession fromClean = TrainingFixture.StartSession(Optimizers.GradientDescent);

        await fromClean.TrainSampleAsync(TrainingFixture.Input, TrainingFixture.Target);

        TrainingFixture.AssertNetworksAgree(fromClean.Network, afterwards.Network);
    }

    [TestMethod]
    public async Task ABatchCancelledPartwayThroughALayerTreeLeavesNoGradientsBehindAsync()
    {
        // A one-layer network never backpropagates to a preceding layer, so it never starts the concurrent work
        //  a cancelled batch has to finish before its gradients are discarded.  Two layers do:  the second
        //  layer's accumulation runs beside the first layer's, and whichever ends first must not clear an array
        //  the other is still writing into.
        using CancellationTokenSource cancellation = new();

        TrainingSession cancelled = TrainingSession.Adopt(
            TrainingFixture.BuildTwoLayerNetwork(),
            TrainingFixture.CreateSettings(
                Optimizers.GradientDescent,
                costFunction: new CancellingCostFunction(cancellation)));

        IReadOnlyList<IReadOnlyList<float>> input = [TrainingFixture.Input, [0.9f, 0.2f]];
        IReadOnlyList<IReadOnlyList<float>> targets = [TrainingFixture.Target, [-0.4f]];

        await Assert.ThrowsExactlyAsync<OperationCanceledException>(
            async () => await cancelled.TrainBatchAsync(input, targets, cancellation.Token));

        // Trained afterwards, the network must move as though the cancelled batch had never accumulated anything.
        TrainingSession afterwards = TrainingSession.Adopt(
            cancelled.Network,
            TrainingFixture.CreateSettings(Optimizers.GradientDescent));

        await afterwards.TrainBatchAsync(input, targets);

        TrainingSession fromClean = TrainingSession.Adopt(
            TrainingFixture.BuildTwoLayerNetwork(),
            TrainingFixture.CreateSettings(Optimizers.GradientDescent));

        await fromClean.TrainBatchAsync(input, targets);

        TrainingFixture.AssertNetworksAgree(fromClean.Network, afterwards.Network);
    }

    [TestMethod]
    public async Task ABatchRefusedBeforeAnythingMovedLeavesTheSessionFitToTrainAsync()
    {
        // The case this exists for:  a gradient no limit can scale is refused in the first of the three passes
        //  the step makes, so nothing has moved, and the session must be able to carry on.  A session that
        //  marked itself unusable on any failure out of the step would pass every assertion above this one and
        //  fail here, which is exactly what it did until 2026-10-13.
        TrainingSession session = TrainingSession.Start(
            TrainingFixture.BuildNeuronNetwork(),
            TrainingFixture.CreateSettings(Optimizers.Adam(), maxGradientNorm: 0.5f));

        InvalidOperationException refusal = await Assert.ThrowsExactlyAsync<InvalidOperationException>(
            async () => await session.TrainSampleAsync(TrainingFixture.Input, [float.NaN]));

        StringAssert.Contains(refusal.Message, "cannot be applied", "The refusal says the limit did not apply.");

        // The same session, the same optimizer state, a sample it can learn from:  it trains, and it trains as
        //  a session that has taken no step, since the refused batch took none.
        await session.TrainSampleAsync(TrainingFixture.Input, TrainingFixture.Target);

        TrainingSession fromClean = TrainingSession.Start(
            TrainingFixture.BuildNeuronNetwork(),
            TrainingFixture.CreateSettings(Optimizers.Adam(), maxGradientNorm: 0.5f));

        await fromClean.TrainSampleAsync(TrainingFixture.Input, TrainingFixture.Target);

        TrainingFixture.AssertNetworksAgree(fromClean.Network, session.Network);
    }

    [TestMethod]
    public async Task AStepThatFailsPartwayLeavesTheGroupsItMovedMovedAsync()
    {
        // Parameters move group by group, so a step that fails on a later group has already moved some of them,
        //  and the session cannot put them back.  What it owes is the gradients discarded and an honest account:
        //  the earlier groups moved, the later ones did not, and training again is the caller's decision rather
        //  than something the session refuses.  Only an optimizer of the caller's own can fail this way.
        INeuralNetwork network = TrainingFixture.BuildTwoLayerNetwork();

        TrainingSession session = TrainingSession.Adopt(
            network,
            TrainingFixture.CreateSettings(ThrowingOptimizer.OnThirdGroup()));

        NeuralLayerDocument before = TrainingFixture.ReadLayers(network)[0];

        await Assert.ThrowsExactlyAsync<InvalidOperationException>(
            async () => await session.TrainSampleAsync(TrainingFixture.Input, TrainingFixture.Target));

        // The first layer holds the first three groups, of which two moved before the third threw.
        NeuralLayerDocument after = TrainingFixture.ReadLayers(network)[0];

        Assert.AreNotEqual(
            before.BiasVector[0],
            after.BiasVector[0],
            "The first group did not move, so this no longer tests a step that failed partway.");

        // Training again is allowed rather than refused, which is the behaviour this replaced:  the optimizer
        //  fails on one call only, so the next sample trains and moves the network on from where it stands.
        await session.TrainSampleAsync(TrainingFixture.Input, TrainingFixture.Target);

        NeuralLayerDocument afterTrainingAgain = TrainingFixture.ReadLayers(network)[0];

        Assert.AreNotEqual(
            after.BiasVector[0],
            afterTrainingAgain.BiasVector[0],
            "The session refused to train after a step that failed partway.");
    }

    [TestMethod]
    public async Task AGradientThatIsNotANumberIsRefusedRatherThanPassingThroughTheLimitAsync()
    {
        // A norm that is not a number compares false against the maximum, so the limit silently did not apply
        //  and the value reached every parameter and the optimizer's state with it.
        TrainingSession session = TrainingSession.Start(
            TrainingFixture.BuildNeuronNetwork(),
            TrainingFixture.CreateSettings(Optimizers.Adam(), maxGradientNorm: 1f));

        NeuralLayerDocument before = TrainingFixture.ReadLayer(session.Network);

        await Assert.ThrowsExactlyAsync<InvalidOperationException>(
            async () => await session.TrainSampleAsync(TrainingFixture.Input, [float.NaN]));

        NetworkAssert.ValuesAre(
            TrainingFixture.ReadLayer(session.Network).WeightMatrix.Single(),
            before.WeightMatrix.Single().Select(weight => (double)weight),
            "weight after the refused batch of");

        NetworkAssert.ValuesAre(
            TrainingFixture.ReadLayer(session.Network).BiasVector,
            before.BiasVector.Select(bias => (double)bias),
            "bias after the refused batch of");
    }

    [TestMethod]
    public async Task AGradientThatIsInfiniteIsRefusedRatherThanScaledToNothingAsync()
    {
        // The other half of a norm no factor can bring within the maximum:  the maximum over infinity is zero,
        //  which would move nothing while reporting a step taken.
        TrainingSession session = TrainingSession.Start(
            TrainingFixture.BuildNeuronNetwork(),
            TrainingFixture.CreateSettings(
                Optimizers.GradientDescent,
                costFunction: new InfiniteCostFunction(),
                maxGradientNorm: 1f));

        NeuralLayerDocument before = TrainingFixture.ReadLayer(session.Network);

        InvalidOperationException refusal = await Assert.ThrowsExactlyAsync<InvalidOperationException>(
            async () => await session.TrainSampleAsync(TrainingFixture.Input, TrainingFixture.Target));

        StringAssert.Contains(refusal.Message, "cannot be applied", "The message says the limit did not apply.");

        NetworkAssert.ValuesAre(
            TrainingFixture.ReadLayer(session.Network).WeightMatrix.Single(),
            before.WeightMatrix.Single().Select(weight => (double)weight),
            "weight after the refused batch of");
    }

    [TestMethod]
    public async Task AnEmptyParameterGroupCarriesTheStateItsOptimizerKeepsAndSurvivesASaveAsync()
    {
        // A group of no parameters still gets state from an optimizer that keeps state, and that state still has
        //  to reach the document and come back:  a state of zero moments, for the masked neuron's empty group.
        TrainingSession session = TrainingSession.Start(
            BuildNetworkWithAMaskedNeuron(),
            TrainingFixture.CreateSettings(Optimizers.Adam(), maxGradientNorm: 0.5f));

        await session.TrainBatchAsync(
            [TrainingFixture.Input, [0.9f, 0.2f]],
            [[0.25f, -0.4f], [-0.1f, 0.3f]]);

        string document = session.ToJson();

        // The network's groups are the bias vector, then one weight group per output value;  the second output
        //  value is the masked one, so its group is empty and is the last of the three.
        JsonArray savedState = JsonNode.Parse(document)!["OptimizerState"]!.AsArray();

        Assert.AreEqual(3, savedState.Count, "One state for each of the network's parameter groups.");

        JsonNode maskedGroupState = savedState[2]!;

        Assert.AreEqual(
            0,
            maskedGroupState["FirstMoments"]!.AsArray().Count,
            "The empty group's state describes no parameters.");

        Assert.AreEqual(
            1,
            maskedGroupState["StepCount"]!.GetValue<int>(),
            "The empty group's optimizer took the batch's step with every other group.");

        // The comparison that has power:  the document resumed against the session continued in memory.  Both
        //  sides train the same sample, so they agree only if the document carried what the session held --
        //  where resuming the same document twice would agree whatever it carried.
        TrainingSession resumed = TrainingSession.Resume(document);

        await resumed.TrainSampleAsync(TrainingFixture.Input, [0.25f, -0.4f]);
        await session.TrainSampleAsync(TrainingFixture.Input, [0.25f, -0.4f]);

        TrainingFixture.AssertNetworksAgree(session.Network, resumed.Network);
    }

    [TestMethod]
    public async Task AnEpochsCostIsTheMeanOverItsSamplesRatherThanOverItsBatchesAsync()
    {
        // Three samples in batches of two leaves a final batch of one, which must count for half as much as the
        //  full batch rather than the same:  the mean over batches would weight that one sample twice over.
        TrainingSession session = TrainingFixture.StartSession(Optimizers.GradientDescent);

        IReadOnlyList<IReadOnlyList<float>> input = [TrainingFixture.Input, [0.9f, 0.2f], [-0.5f, 0.4f]];
        IReadOnlyList<IReadOnlyList<float>> targets = [TrainingFixture.Target, [-0.4f], [0.8f]];

        ReferenceNeuron neuron = TrainingFixture.CreateReferenceNeuron();

        // The first batch's two samples are measured against the starting parameters, and the last sample
        //  against the parameters that batch left behind.
        double firstBatchCost = 0;

        for (int sampleIndex = 0; sampleIndex < 2; sampleIndex++)
            firstBatchCost += CalculateCost(neuron, input[sampleIndex], targets[sampleIndex][0]);

        firstBatchCost /= 2;

        (IReadOnlyList<double> weightGradients, double biasGradient) = neuron.CalculateBatchGradients(
            [.. input.Take(2).Select(sample => (IReadOnlyList<double>)[.. sample.Select(value => (double)value)])],
            [.. targets.Take(2).Select(target => (double)target[0])]);

        for (int index = 0; index < neuron.Weights.Length; index++)
            neuron.Weights[index] -= TrainingFixture.LearningRate * weightGradients[index];

        neuron.Bias -= TrainingFixture.LearningRate * biasGradient;

        double lastBatchCost = CalculateCost(neuron, input[2], targets[2][0]);

        double expectedCost = ((firstBatchCost * 2) + lastBatchCost) / 3;

        float reportedCost = await session.TrainEpochAsync(input, targets, batchSize: 2);

        Assert.AreEqual(expectedCost, reportedCost, NetworkAssert.Tolerance, "The epoch's cost.");
    }

    [TestMethod]
    public async Task ANeuronWhoseConnectionsAreAllMaskedTrainsThroughItsBiasAsync()
    {
        // Such a neuron's group of weights is empty, and the bias is the one parameter it has:  the network has
        //  no input to that neuron, so its output is the activation of the bias alone, and the bias moves by the
        //  gradient that follows from it.
        TrainingSession session = TrainingSession.Start(
            BuildNetworkWithAMaskedNeuron(),
            TrainingFixture.CreateSettings(Optimizers.GradientDescent));

        NeuralLayerDocument before = TrainingFixture.ReadLayer(session.Network);

        float target = -0.4f;

        await session.TrainSampleAsync(TrainingFixture.Input, [0.25f, target]);

        double bias = before.BiasVector[1];
        double output = Math.Tanh(bias);
        double expectedBias = bias - (TrainingFixture.LearningRate * (output - target) * (1 - (output * output)));

        NeuralLayerDocument after = TrainingFixture.ReadLayer(session.Network);

        NetworkAssert.ValuesAre([after.BiasVector[1]], [expectedBias], "masked neuron's bias");

        Assert.AreEqual(0, after.WeightMatrix[1].Count, "The masked neuron has weights.");
    }

    [TestMethod]
    public async Task TrainingReportsTheCostOfWhatItLearnedFromAsync()
    {
        // The cost is what a caller watches to see whether training is working, and it is measured against the
        // parameters as they were when the samples were seen rather than after they moved.
        TrainingSession session = TrainingFixture.StartSession(Optimizers.GradientDescent);

        IReadOnlyList<IReadOnlyList<float>> input = [TrainingFixture.Input, [0.9f, 0.2f]];
        IReadOnlyList<IReadOnlyList<float>> targets = [TrainingFixture.Target, [-0.4f]];

        ReferenceNeuron neuron = TrainingFixture.CreateReferenceNeuron();

        double expectedCost = 0;

        for (int sampleIndex = 0; sampleIndex < input.Count; sampleIndex++)
        {
            double output = neuron.CalculateOutput([.. input[sampleIndex].Select(value => (double)value)]);
            double residual = targets[sampleIndex][0] - output;

            expectedCost += residual * residual / 2;
        }

        expectedCost /= input.Count;

        float reportedCost = await session.TrainBatchAsync(input, targets);

        Assert.AreEqual(expectedCost, reportedCost, NetworkAssert.Tolerance, "The batch's cost.");

        // Training the same batch again from the moved parameters must cost less:  the step was taken to
        // reduce exactly this.
        TrainingSession fresh = TrainingFixture.StartSession(Optimizers.GradientDescent);

        float firstCost = await fresh.TrainBatchAsync(input, targets);
        float secondCost = await fresh.TrainBatchAsync(input, targets);

        Assert.IsTrue(secondCost < firstCost, $"The cost did not fall ({firstCost} then {secondCost}).");
    }

    #endregion

    #region Private Static

    /// <summary>
    /// Builds a sparse layer of two neurons, the second of which has no connections at all.
    /// </summary>
    /// <returns>
    /// Returns the network, whose second neuron trains through its bias alone.
    /// </returns>
    private static INeuralNetwork BuildNetworkWithAMaskedNeuron()
    {
        ConnectionMaskBuilder maskBuilder = new(inputCount: 2, outputCount: 2);

        maskBuilder.MaskAllConnections(neuronIndex: 1);

        NeuralNetworkBuilder builder = new(inputCount: 2);

        builder.AddSparseLayer(
            maskBuilder.Build(),
            ActivationFunctions.Tanh,
            new GlorotUniformWeightInitializer(seed: 5));

        return builder.BuildNetwork();
    }

    /// <summary>
    /// Calculates what the mean squared error makes of one sample, in double precision.
    /// </summary>
    /// <param name="neuron">
    /// The neuron as it stands before the sample.
    /// </param>
    /// <param name="input">
    /// The sample's input values.
    /// </param>
    /// <param name="target">
    /// The sample's desired output value.
    /// </param>
    /// <returns>
    /// Returns the sample's cost.
    /// </returns>
    private static double CalculateCost(ReferenceNeuron neuron, IReadOnlyList<float> input, double target)
    {
        double residual = target - neuron.CalculateOutput([.. input.Select(value => (double)value)]);

        return residual * residual / 2;
    }

    #endregion

    #region Private

    /// <summary>
    /// A cost that cancels the batch once a sample has been learned from.
    /// </summary>
    /// <remarks>
    /// The derivative is asked for once per output value per sample, so cancelling on the second call abandons
    /// the batch with one sample's gradients already accumulated.
    /// </remarks>
    private sealed class CancellingCostFunction : ICostFunction
    {
        public CancellingCostFunction(CancellationTokenSource cancellation)
        {
            cancellation.ThrowIfNull();

            Cancellation = cancellation;
        }

        public string ShortName => "CancelsAfterOneSample";

        public float Calculate(float targetValue, float actualValue) =>
            CostFunctions.MeanSquaredError.Calculate(targetValue, actualValue);

        public float CalculateDerivative(float targetValue, float actualValue)
        {
            CallCount += 1;

            if (CallCount == 2)
                Cancellation.Cancel();

            return CostFunctions.MeanSquaredError.CalculateDerivative(targetValue, actualValue);
        }

        private CancellationTokenSource Cancellation { get; }

        private int CallCount { get; set; }
    }

    /// <summary>
    /// A cost whose derivative is infinite, standing for a network whose gradients have overflowed.
    /// </summary>
    private sealed class InfiniteCostFunction : ICostFunction
    {
        public string ShortName => "Infinite";

        public float Calculate(float targetValue, float actualValue) => float.PositiveInfinity;

        public float CalculateDerivative(float targetValue, float actualValue) => float.PositiveInfinity;
    }

    /// <summary>
    /// An optimizer that refuses to take a step, standing for any failure while the parameters are moving.
    /// </summary>
    /// <remarks>
    /// Which call it fails on is the point:  failing on the first group leaves the network untouched, and failing
    /// on a later one leaves the groups before it moved, which is the case a session cannot undo.
    /// </remarks>
    private sealed class ThrowingOptimizer : IOptimizer
    {
        private ThrowingOptimizer(int failingCall)
        {
            FailingCall = failingCall;
        }

        /// <summary>
        /// An optimizer that fails before it has moved anything.
        /// </summary>
        /// <returns>
        /// Returns the optimizer.
        /// </returns>
        public static ThrowingOptimizer OnFirstGroup() => new(failingCall: 1);

        /// <summary>
        /// An optimizer that moves two groups and then fails, which a two-layer network reaches.
        /// </summary>
        /// <returns>
        /// Returns the optimizer.
        /// </returns>
        public static ThrowingOptimizer OnThirdGroup() => new(failingCall: 3);

        public string ShortName => "Throwing";

        public string LongName => "Throws Instead Of Stepping";

        public IOptimizerState? CreateState(int parameterCount) => null;

        public bool CanApplyTo(IOptimizerState? state, int parameterCount) => state is null;

        public void ApplyGradients(
            IOptimizerState? state,
            float[] parameters,
            ReadOnlySpan<float> gradients,
            float learningRate)
        {
            CallCount += 1;

            if (CallCount == FailingCall)
                throw new InvalidOperationException($"This optimizer fails on call {FailingCall}.");

            for (int index = 0; index < parameters.Length; index++)
                parameters[index] -= learningRate * gradients[index];
        }

        private int FailingCall { get; }

        private int CallCount { get; set; }
    }

    #endregion
}
