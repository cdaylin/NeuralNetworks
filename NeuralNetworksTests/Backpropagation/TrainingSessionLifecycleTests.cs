using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.Backpropagation.Optimization;
using Daylin.NeuralNetworks.Serialization;
using Daylin.NeuralNetworks.WeightInitialization;

using System.Text.Json.Nodes;

namespace Daylin.NeuralNetworks.Backpropagation;

/// <summary>
/// Tests what a session does with the network it is given and the state it is resumed from.
/// </summary>
/// <remarks>
/// A session owns the optimizer's state while a layer holds it, so the questions here are about ownership:
/// which network a session trains, what it starts from, what it refuses to resume from, and what a saved
/// network carries.
/// </remarks>
[TestClass]
public sealed class TrainingSessionLifecycleTests
{
    #region Public

    [TestMethod]
    public async Task AdoptingANetworkTrainedByAnEarlierSessionStartsTheOptimizerAfreshAsync()
    {
        // The network's layers hold the optimizer's state, and an adopted network keeps holding it, so a new
        //  session must discard it:  a session that has taken no steps must move as one that has taken none.
        //  The optimizer is the same kind in both sessions, which is the case the state would survive.
        INeuralNetwork network = TrainingFixture.BuildNeuronNetwork();

        TrainingSession firstSession = TrainingSession.Adopt(
            network,
            TrainingFixture.CreateSettings(Optimizers.Momentum(momentum: 0.9f)));

        await firstSession.TrainSampleAsync(Input, Target);

        NeuralLayerDocument afterFirstSession = TrainingFixture.ReadLayer(network);

        TrainingSession secondSession = TrainingSession.Adopt(
            network,
            TrainingFixture.CreateSettings(Optimizers.Momentum(momentum: 0.9f)));

        await secondSession.TrainSampleAsync(Input, Target);

        // A velocity carried over from the first session would make this step larger than a first step.
        ReferenceNeuron neuron = new(
            afterFirstSession.WeightMatrix.Single().Select(weight => (double)weight),
            afterFirstSession.BiasVector.Single());

        (IReadOnlyList<double> weightGradients, double biasGradient) =
            neuron.CalculateGradients([.. Input.Select(value => (double)value)], Target[0]);

        for (int index = 0; index < neuron.Weights.Length; index++)
            neuron.Weights[index] -= TrainingFixture.LearningRate * weightGradients[index];

        neuron.Bias -= TrainingFixture.LearningRate * biasGradient;

        NeuralLayerDocument layer = TrainingFixture.ReadLayer(network);

        NetworkAssert.ValuesAre(layer.WeightMatrix.Single(), neuron.Weights, "weight of");
        NetworkAssert.ValuesAre(layer.BiasVector, [neuron.Bias], "bias of");
    }

    [TestMethod]
    public async Task ASavedSessionIsNotResumedAgainstANetworkOfAnotherShapeAsync()
    {
        // Two networks with the same number of parameter groups and differently sized groups:  a session that
        //  compared only the number of groups would import this state and then discard it at the first step,
        //  which is the quiet restart the refusal exists to prevent.
        JsonNode overThreeInputs = await SaveTrainedSessionAsync(inputCount: 3);
        JsonNode overTwoInputs = await SaveTrainedSessionAsync(inputCount: 2);

        Assert.AreEqual(
            overThreeInputs["OptimizerState"]!.AsArray().Count,
            overTwoInputs["OptimizerState"]!.AsArray().Count,
            "The two networks hold a different number of parameter groups, so this tests the wrong thing.");

        overTwoInputs["OptimizerState"] = overThreeInputs["OptimizerState"]!.DeepClone();

        ArgumentException exception = Assert.ThrowsExactly<ArgumentException>(
            () => TrainingSession.Resume(overTwoInputs.ToJsonString()));

        // The group that disagrees is the weights, which follow the bias:  naming it is what distinguishes this
        //  refusal from the one a mismatched number of groups gives.
        StringAssert.Contains(exception.Message, "parameter group 1", "The message names the group that disagrees.");
    }

    [TestMethod]
    public void StartRefusesANetworkTheLibraryDidNotBuild()
    {
        // Asserting on the message is what pins the order:  a Start that copied the network before testing its
        //  type would be refused by the serializer instead, in the serializer's own terms.
        ArgumentException exception = Assert.ThrowsExactly<ArgumentException>(
            () => TrainingSession.Start(
                new ForeignNetwork(),
                TrainingFixture.CreateSettings(Optimizers.GradientDescent)));

        StringAssert.Contains(exception.Message, nameof(ForeignNetwork), "The message names the network's type.");
    }

    [TestMethod]
    public void AdoptRefusesANetworkTheLibraryDidNotBuild()
    {
        ArgumentException exception = Assert.ThrowsExactly<ArgumentException>(
            () => TrainingSession.Adopt(
                new ForeignNetwork(),
                TrainingFixture.CreateSettings(Optimizers.GradientDescent)));

        StringAssert.Contains(exception.Message, nameof(ForeignNetwork), "The message names the network's type.");
    }

    [TestMethod]
    public async Task TrainingRejectsSamplesThatAreNotThereAsync()
    {
        TrainingSession session = TrainingFixture.StartSession(Optimizers.GradientDescent);

        await Assert.ThrowsExactlyAsync<ArgumentNullException>(
            async () => await session.TrainSampleAsync(null!, Target));

        await Assert.ThrowsExactlyAsync<ArgumentNullException>(
            async () => await session.TrainSampleAsync(Input, null!));

        await Assert.ThrowsExactlyAsync<ArgumentNullException>(
            async () => await session.TrainBatchAsync(null!, [Target]));

        await Assert.ThrowsExactlyAsync<ArgumentNullException>(
            async () => await session.TrainEpochAsync([Input], null!, batchSize: 1));
    }

    [TestMethod]
    public async Task ASavedNetworkCarriesNoPropertyOfTrainingAsync()
    {
        // Asserted against the document's property names rather than a search for "Adam", which only the
        //  optimizer the test happens to use would put there.
        TrainingSession session = TrainingFixture.StartSession(Optimizers.Adam());

        await session.TrainSampleAsync(Input, Target);

        JsonNode document = JsonNode.Parse(NetworkDocument.FromNetwork(session.Network).ToJson())!;

        foreach (string name in TrainingPropertyNames)
            Assert.IsFalse(HasProperty(document, name), $"The network document carries '{name}'.");
    }

    [TestMethod]
    public async Task ASavedSessionCarriesThePropertiesItIsResumedFromAsync()
    {
        // The other half of the assertion above:  the properties a network document must not carry are the ones
        //  a session document must, so a test that only looked for their absence would pass on a session that
        //  saved nothing.
        TrainingSession session = TrainingFixture.StartSession(Optimizers.Adam());

        await session.TrainSampleAsync(Input, Target);

        JsonNode document = JsonNode.Parse(session.ToJson())!;

        foreach (string name in TrainingPropertyNames)
            Assert.IsTrue(HasProperty(document, name), $"The session document does not carry '{name}'.");
    }

    #endregion

    #region Private Static

    /// <summary>
    /// Property names that belong to a session's document and never to a network's.
    /// </summary>
    private static IReadOnlyList<string> TrainingPropertyNames { get; } =
        ["OptimizerState", "CompletedEpochCount", "FirstMoments", "SecondMoments", "StepCount"];

    private static IReadOnlyList<float> Input => TrainingFixture.Input;

    private static IReadOnlyList<float> Target => TrainingFixture.Target;

    /// <summary>
    /// Trains a one-neuron network over a given number of input values, and saves the session it trained in.
    /// </summary>
    /// <param name="inputCount">
    /// Number of input values, which is the size of the network's one group of weights.
    /// </param>
    /// <returns>
    /// Returns the saved session's document.
    /// </returns>
    private static async Task<JsonNode> SaveTrainedSessionAsync(int inputCount)
    {
        TrainingSession session = TrainingSession.Start(
            TrainingFixture.BuildNeuronNetwork(inputCount),
            TrainingFixture.CreateSettings(Optimizers.Adam()));

        await session.TrainSampleAsync([.. Enumerable.Range(1, inputCount).Select(value => (float)value)], Target);

        return JsonNode.Parse(session.ToJson())!;
    }

    private static bool HasProperty(JsonNode node, string name) =>
        node switch
        {
            JsonObject properties => properties.Any(property =>
                property.Key == name || (property.Value is not null && HasProperty(property.Value, name))),

            JsonArray values => values.Any(value => value is not null && HasProperty(value, name)),

            _ => false
        };

    #endregion

    #region Private

    /// <summary>
    /// A network implemented outside the library, which the public interface allows and a session cannot train.
    /// </summary>
    private sealed class ForeignNetwork : INeuralNetwork
    {
        public string ShortDescription => "Foreign";

        public int InputCount => 2;

        public int OutputCount => 1;

        public Task<IEnumerable<float>> CalculateOutputAsync(
            IEnumerable<float> input,
            CancellationToken cancellationToken = default)
        {
            return Task.FromResult<IEnumerable<float>>([0f]);
        }
    }

    #endregion
}
