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
            CostDerivatives.MeanSquaredError,
            Optimizers.Adam(firstMomentDecay: 0.8f, secondMomentDecay: 0.99f, epsilon: 1e-7f),
            LearningRateSchedulers.ExponentialDecayLearningRateScheduler(0.25f, decayRate: 0.9f));

        string document = JsonSerializer.Serialize(settings);

        // A configuration is worth writing down on its own:  no network, and no session's progress.
        Assert.IsFalse(document.Contains("NetworkLayer", StringComparison.Ordinal), "A network in the document.");
        Assert.IsFalse(document.Contains("Epoch", StringComparison.Ordinal), "A session's progress.");

        TrainingSettings restored = JsonSerializer.Deserialize<TrainingSettings>(document)!;

        Assert.AreEqual(settings.CostDerivative.ShortName, restored.CostDerivative.ShortName, "Cost.");
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
    }

    #endregion
}
