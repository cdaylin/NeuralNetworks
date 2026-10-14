using Daylin.NeuralNetworks.Backpropagation.Cost;
using Daylin.NeuralNetworks.Backpropagation.LearningRate;
using Daylin.NeuralNetworks.Backpropagation.Optimization;
using Daylin.TestUtilities;

using System.Text.Json;

namespace Daylin.NeuralNetworks.Backpropagation;

/// <summary>
/// Tests the schedulers that supply a learning rate for each training epoch.
/// </summary>
[TestClass]
public class LearningRateSchedulerTests
{
    #region Public

    [TestMethod]
    public void AConstantSchedulerGivesOneRateForEveryEpoch()
    {
        ILearningRateScheduler scheduler = LearningRateSchedulers.ConstantLearningRateScheduler(0.25f);

        Assert.AreEqual(0.25f, scheduler.GetLearningRate(epochCount: 0));
        Assert.AreEqual(0.25f, scheduler.GetLearningRate(epochCount: 10));
    }

    [TestMethod]
    public void AnExponentialSchedulerKeepsOneFractionOfTheRateEachEpoch()
    {
        ILearningRateScheduler scheduler =
            LearningRateSchedulers.ExponentialDecayLearningRateScheduler(0.8f, decayRate: 0.5f);

        // The first epoch trains at the initial rate, since no epoch has decayed it yet.
        NetworkAssert.ValuesAre(
            [
                scheduler.GetLearningRate(epochCount: 0),
                scheduler.GetLearningRate(epochCount: 1),
                scheduler.GetLearningRate(epochCount: 2),
                scheduler.GetLearningRate(epochCount: 3),
            ],
            [0.8, 0.4, 0.2, 0.1],
            "learning rate for epoch");
    }

    [TestMethod]
    public void AnExponentialSchedulerThatDecaysByNothingGivesOneRateForEveryEpoch()
    {
        ILearningRateScheduler scheduler =
            LearningRateSchedulers.ExponentialDecayLearningRateScheduler(0.25f, decayRate: 1);

        Assert.AreEqual(0.25f, scheduler.GetLearningRate(epochCount: 0));
        Assert.AreEqual(0.25f, scheduler.GetLearningRate(epochCount: 10));
    }

    [TestMethod]
    public void TrainingSettingsSurviveBeingSavedOnTheirOwn()
    {
        TrainingSettings settings = new(
            CostFunctions.MeanSquaredError,
            Optimizers.Adam(firstMomentDecay: 0.8f, secondMomentDecay: 0.99f, epsilon: 1e-7f),
            LearningRateSchedulers.ExponentialDecayLearningRateScheduler(0.25f, decayRate: 0.9f));

        string document = JsonSerializer.Serialize(settings);

        // A configuration is worth writing down on its own:  no network, and no session's progress.
        Assert.IsFalse(document.Contains("NetworkLayer", StringComparison.Ordinal), "A network in the document.");
        Assert.IsFalse(document.Contains("Epoch", StringComparison.Ordinal), "A session's progress.");

        TrainingSettings restored = JsonSerializer.Deserialize<TrainingSettings>(document)!;

        Assert.AreEqual(settings.CostFunction.ShortName, restored.CostFunction.ShortName, "Cost.");
        Assert.AreEqual(settings.Optimizer.ShortName, restored.Optimizer.ShortName, "Optimizer.");

        AdamOptimizer adam = (AdamOptimizer)restored.Optimizer;

        Assert.AreEqual(0.8f, adam.FirstMomentDecay, "Adam's first moment decay.");
        Assert.AreEqual(0.99f, adam.SecondMomentDecay, "Adam's second moment decay.");
        Assert.AreEqual(1e-7f, adam.Epsilon, "Adam's epsilon.");

        NetworkAssert.ValuesAre(
            [restored.LearningRateScheduler.GetLearningRate(0), restored.LearningRateScheduler.GetLearningRate(2)],
            [0.25, 0.25 * 0.9 * 0.9],
            "restored schedule's learning rate for epoch");
    }

    [TestMethod]
    public void ASchedulerRejectsARateOrADecayOutsideItsRange()
    {
        Asserter.AssertThat(() => LearningRateSchedulers.ExponentialDecayLearningRateScheduler(0, 0.5f))
            .Throws<ArgumentOutOfRangeException>();

        Asserter.AssertThat(() => LearningRateSchedulers.ExponentialDecayLearningRateScheduler(0.5f, 0))
            .Throws<ArgumentOutOfRangeException>();

        Asserter.AssertThat(() => LearningRateSchedulers.ExponentialDecayLearningRateScheduler(0.5f, 1.5f))
            .Throws<ArgumentOutOfRangeException>();

        // A rate of zero would run every batch, count every epoch, advance the optimizer's state, and move no
        //  parameter, while the interface documents a positive rate.
        Asserter.AssertThat(() => LearningRateSchedulers.ConstantLearningRateScheduler(0))
            .Throws<ArgumentOutOfRangeException>();

        Asserter.AssertThat(() => LearningRateSchedulers.ConstantLearningRateScheduler(-0.1f))
            .Throws<ArgumentOutOfRangeException>();
    }

    [TestMethod]
    public async Task ASessionRefusesARateItsScheduleReturnsThatWouldTrainNothingAsync()
    {
        // A schedule is free to return a rate it was never constructed with -- an exponential decay underflows
        //  to zero after enough epochs -- so the rate is checked where it is read.  A schedule of the test's own
        //  is what lets this be tested without training for nine hundred epochs;  it is handed to the session
        //  directly, since an unregistered schedule has no discriminator and so cannot reach a document.
        TrainingSettings settings = new(
            CostFunctions.MeanSquaredError,
            Optimizers.GradientDescent,
            new ExhaustedScheduler());

        TrainingSession session = TrainingSession.Adopt(TrainingFixture.BuildNeuronNetwork(), settings);

        InvalidOperationException refusal = await Assert.ThrowsExactlyAsync<InvalidOperationException>(
            async () => await session.TrainSampleAsync(TrainingFixture.Input, TrainingFixture.Target));

        StringAssert.Contains(
            refusal.Message,
            nameof(ExhaustedScheduler),
            "The refusal names the schedule that returned the rate, rather than a parameter nobody passed.");
    }

    #endregion

    #region Private

    /// <summary>
    /// A schedule whose rate has decayed to nothing, as an exponential decay's does in time.
    /// </summary>
    private sealed class ExhaustedScheduler : ILearningRateScheduler
    {
        public float GetLearningRate(int epochCount) => 0;
    }

    #endregion
}
