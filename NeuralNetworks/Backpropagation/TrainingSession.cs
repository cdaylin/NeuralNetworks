using Daylin.NeuralNetworks.Backpropagation.Optimization;

using System.Text.Json;
using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Backpropagation;

/// <summary>
/// A course of training:  a network, the settings it is being trained under, and everything training remembers
/// between one batch and the next.
/// </summary>
/// <remarks>
/// A network is a function from input values to output values.  How it came to hold the weights it holds --
/// which optimizer moved them, what that optimizer remembers, how many epochs have passed -- is <em>about</em>
/// the network rather than part of it, and lives here.  A network saved on its own is therefore what it always
/// was, and a session saved on its own can be resumed exactly.
/// <para>
/// A session must be the only thing training its network, because its optimizer's state describes those
/// weights and is meaningless against weights something else has moved.  There are two ways to have that, and
/// the choice is the caller's:  <see cref="Start"/> gives the session a copy, and <see cref="Adopt"/> gives it
/// the caller's own network on the caller's guarantee that nothing else will train it.
/// </para><para>
/// The settings are the session's own, recorded when it started and restored when it resumes, so a session
/// cannot be continued against a cost function, an optimizer, or a schedule other than the one whose state it
/// carries.
/// </para>
/// </remarks>
public sealed class TrainingSession
{
    #region Public Static

    /// <summary>
    /// Starts a session training a copy of a network, leaving the caller's network as it is.
    /// </summary>
    /// <remarks>
    /// The copy is what makes this safe whatever else holds the network:  several sessions may be started from
    /// one network, and none of them can disturb another or the network they were started from.
    /// <para>
    /// It is also what it costs.  Copying is a serialization round trip, so its time and the memory it takes
    /// both grow with the network's parameter count, and the copy lives as long as the session does.  A caller
    /// starting many sessions, or minding memory, can pay neither by using <see cref="Adopt"/> instead.
    /// </para>
    /// </remarks>
    /// <param name="network">
    /// The network to train.  It is copied, and never touched again by the session.
    /// </param>
    /// <param name="settings">
    /// Settings governing the training, which the session keeps for as long as it lasts.
    /// </param>
    /// <returns>
    /// Returns a session that has completed no epochs and whose optimizer has taken no steps.
    /// </returns>
    public static TrainingSession Start(INeuralNetwork network, TrainingSettings settings)
    {
        network.ThrowIfNull();
        settings.ThrowIfNull();

        return new TrainingSession(Copy(network), settings, completedEpochCount: 0, optimizerState: null);
    }

    /// <summary>
    /// Starts a session training the caller's own network, without copying it.
    /// </summary>
    /// <remarks>
    /// The session trains the network the caller passed, so the caller sees the weights move and pays nothing
    /// for a copy.  In exchange the caller promises that nothing else trains that network while the session
    /// lasts -- no second session over the same network.
    /// <para>
    /// Nothing enforces the promise, and breaking it is quiet.  Two sessions adopting one network interleave
    /// their steps, after which each one's optimizer state describes weights the other moved:  momentum
    /// carries velocities from steps it did not take, Adam's moments no longer estimate anything, and training
    /// grows worse in a way no exception reports.  Where a caller cannot make that promise,
    /// <see cref="Start"/> is the one that does not need it.
    /// </para><para>
    /// A session saved and resumed is never adopted:  <see cref="Resume"/> builds its network from the
    /// document, so it owns that network outright.
    /// </para>
    /// </remarks>
    /// <param name="network">
    /// The network to train, which the session trains in place.
    /// </param>
    /// <param name="settings">
    /// Settings governing the training, which the session keeps for as long as it lasts.
    /// </param>
    /// <returns>
    /// Returns a session that has completed no epochs and whose optimizer has taken no steps.
    /// </returns>
    public static TrainingSession Adopt(INeuralNetwork network, TrainingSettings settings)
    {
        network.ThrowIfNull();
        settings.ThrowIfNull();

        return new TrainingSession(network, settings, completedEpochCount: 0, optimizerState: null);
    }

    /// <summary>
    /// Resumes a session from the JSON it was saved to.
    /// </summary>
    /// <param name="json">
    /// JSON holding a saved session.
    /// </param>
    /// <returns>
    /// Returns the session, carrying the network, settings, optimizer state and epoch count it was saved with.
    /// </returns>
    public static TrainingSession Resume(string json)
    {
        json.ThrowIfNullOrEmptyOrWhitespace();

        return JsonSerializer.Deserialize<TrainingSession>(json)
            ?? throw new ArgumentException("The JSON holds no training session.", nameof(json));
    }

    #endregion

    #region Private Static

    /// <summary>
    /// Copies a network, through the JSON it serializes to.
    /// </summary>
    /// <param name="network">
    /// The network to copy.
    /// </param>
    /// <returns>
    /// Returns a network holding the same weights, sharing nothing with the one it was copied from.
    /// </returns>
    private static NeuralNetwork Copy(INeuralNetwork network)
    {
        INeuralNetwork? copy = JsonSerializer.Deserialize<INeuralNetwork>(JsonSerializer.Serialize(network));

        return copy as NeuralNetwork
            ?? throw new ArgumentException(
                $"A training session trains a network this library built, and '{network.GetType().Name}' "
                + "is not one.",
                nameof(network));
    }

    #endregion

    #region Construction

    [JsonConstructor]
    public TrainingSession(
        INeuralNetwork network,
        TrainingSettings settings,
        int completedEpochCount,
        IReadOnlyList<IOptimizerState?>? optimizerState)
    {
        network.ThrowIfNull();
        settings.ThrowIfNull();
        completedEpochCount.ThrowIfNegative();

        TrainedNetwork = network as NeuralNetwork
            ?? throw new ArgumentException(
                $"A training session trains a network this library built, and '{network.GetType().Name}' "
                + "is not one.",
                nameof(network));

        Settings = settings;
        CompletedEpochCount = completedEpochCount;

        if (optimizerState is not null)
            RestoreOptimizerState(optimizerState);
    }

    #endregion

    #region Public

    public override string ToString() =>
        $"{TrainedNetwork.ShortDescription} ({Settings.Optimizer.ShortName}, {CompletedEpochCount} epochs)";

    /// <summary>
    /// The network the session is training, and the trained network it produces.
    /// </summary>
    /// <remarks>
    /// The copy the session took, for a session from <see cref="Start"/> or <see cref="Resume"/>; the caller's
    /// own network, for one from <see cref="Adopt"/>.
    /// </remarks>
    [JsonInclude]
    public INeuralNetwork Network => TrainedNetwork;

    /// <summary>
    /// The settings the session was started with, and resumes on.
    /// </summary>
    [JsonInclude]
    public TrainingSettings Settings { get; }

    /// <summary>
    /// Number of training epochs the session has completed.
    /// </summary>
    /// <remarks>
    /// An epoch is one call to <see cref="TrainEpochAsync"/>, and is what the learning rate scheduler reduces
    /// the learning rate over.  Training a single batch completes no epoch.
    /// </remarks>
    [JsonInclude]
    public int CompletedEpochCount { get; private set; }

    /// <summary>
    /// State the session's optimizer is carrying, one entry for each of the network's parameter groups.
    /// </summary>
    /// <remarks>
    /// Written so that a saved session resumes where it left off, and read back through the constructor.  A
    /// session whose optimizer keeps no state, or which has taken no step, carries nothing.
    /// </remarks>
    [JsonInclude]
    public IReadOnlyList<IOptimizerState?>? OptimizerState =>
        TrainedNetwork.ExportOptimizerState() is { } state && state.Any(entry => entry is not null)
            ? state
            : null;

    /// <summary>
    /// Trains the network on one sample, moving its parameters once.
    /// </summary>
    /// <remarks>
    /// One sample is a batch of one, and this is that batch:  the parameters move against the gradients the
    /// sample calls for, by the session's optimizer, at the rate its schedule gives for the epoch in progress.
    /// It completes no epoch.
    /// <para>
    /// This is the training a caller does in a loop of its own, sample by sample as they arrive, where a
    /// caller with a set of samples in hand uses
    /// <see cref="TrainBatchAsync(IReadOnlyList{IReadOnlyList{float}}, IReadOnlyList{IReadOnlyList{float}},
    /// CancellationToken)"/> or <see cref="TrainEpochAsync"/> instead.
    /// </para>
    /// </remarks>
    /// <param name="input">
    /// Input values.  The count must equal the network's input count.
    /// </param>
    /// <param name="targetOutput">
    /// Desired output values.  The count must equal the network's output count.
    /// </param>
    /// <param name="cancellationToken">
    /// A cancellation token.
    /// </param>
    /// <returns>
    /// Returns a task representing this <c>async</c> operation.
    /// </returns>
    public async Task TrainSampleAsync(
        IReadOnlyList<float> input,
        IReadOnlyList<float> targetOutput,
        CancellationToken cancellationToken = default)
    {
        await TrainBatchAsync([input], [targetOutput], cancellationToken);
    }

    /// <summary>
    /// Trains the network on a batch of samples, moving its parameters once.
    /// </summary>
    /// <remarks>
    /// Every sample in the batch is learned from before any parameter moves, and each parameter then moves
    /// against the gradients the samples called for, averaged over the batch.  A batch is the samples it is
    /// given, so a batch of one is <see cref="TrainSampleAsync"/>.
    /// <para>
    /// A batch is all or nothing:  cancelled partway through, it leaves the network as it found it.
    /// </para>
    /// </remarks>
    /// <param name="input">
    /// Input values for each sample in the batch.
    /// </param>
    /// <param name="targetOutput">
    /// Desired output values for each sample in the batch.
    /// </param>
    /// <param name="cancellationToken">
    /// A cancellation token.
    /// </param>
    /// <returns>
    /// Returns a task representing this <c>async</c> operation.
    /// </returns>
    public async Task TrainBatchAsync(
        IReadOnlyList<IReadOnlyList<float>> input,
        IReadOnlyList<IReadOnlyList<float>> targetOutput,
        CancellationToken cancellationToken = default)
    {
        await TrainBatchAsync(
            Settings.LearningRateScheduler.GetLearningRate(CompletedEpochCount),
            input,
            targetOutput,
            cancellationToken);
    }

    /// <summary>
    /// Trains the network on a set of samples, one batch at a time, and completes a training epoch.
    /// </summary>
    /// <remarks>
    /// The samples are taken in the order given, in batches of <paramref name="batchSize"/>, the last batch
    /// holding whatever remains.  The session does not shuffle them:  a caller wanting them shuffled shuffles
    /// them, which keeps training reproducible.
    /// <para>
    /// The learning rate is read from the scheduler once, before the first batch, so every batch in an epoch
    /// trains at the same rate.  <see cref="CompletedEpochCount"/> advances once the last batch is done, so an
    /// epoch abandoned partway through completes no epoch and leaves the batches it did finish behind it.
    /// </para>
    /// </remarks>
    /// <param name="input">
    /// Input values for each sample.
    /// </param>
    /// <param name="targetOutput">
    /// Desired output values for each sample.
    /// </param>
    /// <param name="batchSize">
    /// Maximum number of samples in a batch.
    /// </param>
    /// <param name="cancellationToken">
    /// A cancellation token.
    /// </param>
    /// <returns>
    /// Returns a task representing this <c>async</c> operation.
    /// </returns>
    public async Task TrainEpochAsync(
        IReadOnlyList<IReadOnlyList<float>> input,
        IReadOnlyList<IReadOnlyList<float>> targetOutput,
        int batchSize,
        CancellationToken cancellationToken = default)
    {
        input.ThrowIfEmpty();
        targetOutput.Count.ThrowIfNotEqualTo(input.Count);
        batchSize.ThrowIfNotPositive();

        float learningRate = Settings.LearningRateScheduler.GetLearningRate(CompletedEpochCount);

        for (int firstIndex = 0; firstIndex < input.Count; firstIndex += batchSize)
        {
            int sampleCount = Math.Min(batchSize, input.Count - firstIndex);

            await TrainBatchAsync(
                learningRate,
                Slice(input, firstIndex, sampleCount),
                Slice(targetOutput, firstIndex, sampleCount),
                cancellationToken);
        }

        CompletedEpochCount += 1;
    }

    /// <summary>
    /// Saves the session, including the network it is training.
    /// </summary>
    /// <returns>
    /// Returns JSON that <see cref="Resume(string)"/> restores the session from.
    /// </returns>
    public string ToJson() => JsonSerializer.Serialize(this);

    #endregion

    #region Private

    /// <summary>
    /// The session's own copy of the network, as the type the library builds, so that the session can reach
    /// the training seams the interface does not expose.
    /// </summary>
    private NeuralNetwork TrainedNetwork { get; }

    private static IReadOnlyList<IReadOnlyList<float>> Slice(
        IReadOnlyList<IReadOnlyList<float>> samples,
        int firstIndex,
        int count)
    {
        return samples.Skip(firstIndex).Take(count).ToList();
    }

    /// <summary>
    /// Gives the network the optimizer state a saved session carried, refusing state of another shape.
    /// </summary>
    /// <remarks>
    /// State that does not describe this network's parameters is not state this session can continue from:
    /// the network in the document and the state beside it disagree, and quietly discarding either would make
    /// a resumed session train differently from the one that was saved.
    /// </remarks>
    /// <param name="optimizerState">
    /// One entry for each parameter group, as the session saved them.
    /// </param>
    private void RestoreOptimizerState(IReadOnlyList<IOptimizerState?> optimizerState)
    {
        if (optimizerState.Count != TrainedNetwork.ParameterGroupCount)
        {
            throw new ArgumentException(
                $"The session carries optimizer state for {optimizerState.Count} parameter groups, and its "
                + $"network has {TrainedNetwork.ParameterGroupCount}.",
                nameof(optimizerState));
        }

        TrainedNetwork.ImportOptimizerState(optimizerState);
    }

    /// <summary>
    /// Trains the network on a batch of samples at a learning rate already chosen.
    /// </summary>
    /// <param name="learningRate">
    /// The learning rate for the epoch in progress.
    /// </param>
    /// <param name="input">
    /// Input values for each sample in the batch.
    /// </param>
    /// <param name="targetOutput">
    /// Desired output values for each sample in the batch.
    /// </param>
    /// <param name="cancellationToken">
    /// A cancellation token.
    /// </param>
    /// <returns>
    /// Returns a task representing this <c>async</c> operation.
    /// </returns>
    private async Task TrainBatchAsync(
        float learningRate,
        IReadOnlyList<IReadOnlyList<float>> input,
        IReadOnlyList<IReadOnlyList<float>> targetOutput,
        CancellationToken cancellationToken)
    {
        input.ThrowIfEmpty();
        targetOutput.Count.ThrowIfNotEqualTo(input.Count);

        try
        {
            for (int sampleIndex = 0; sampleIndex < input.Count; sampleIndex++)
            {
                await TrainedNetwork.LearnFromSampleAsync(
                    Settings.CostDerivative,
                    input[sampleIndex],
                    targetOutput[sampleIndex],
                    cancellationToken);
            }

            cancellationToken.ThrowIfCancellationRequested();
        }
        catch
        {
            TrainedNetwork.DiscardGradients();

            throw;
        }

        TrainedNetwork.ApplyGradients(Settings.Optimizer, learningRate, input.Count, Settings.MaxGradientNorm);
    }

    #endregion
}
