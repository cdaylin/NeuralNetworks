using Daylin.NeuralNetworks.Backpropagation.Optimization;

using System.Text.Json;
using System.Text.Json.Nodes;
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
/// </para><para>
/// A session trains one call at a time.  Every training method accumulates gradients into arrays the whole
/// network shares, so two calls overlapping on one session average one another's samples together, move the
/// parameters twice, and leave the optimizer describing weights it did not move -- the same damage as two
/// sessions over one network, and equally quiet.  Awaiting each call before making the next is what a caller
/// owes;  nothing enforces it.
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

        return new TrainingSession(
            Copy(AsTrainedNetwork(network)),
            settings,
            completedEpochCount: 0,
            optimizerState: null);
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
    /// Adopting a network <em>writes</em> to it before any training:  a network an earlier session trained is
    /// still holding that session's optimizer state, and this one discards it, since its own optimizer has taken
    /// no steps.  So adopting a network another session is training does not merely risk interleaved steps -- it
    /// takes that session's state away at once.
    /// </para><para>
    /// A session saved and resumed is never adopted:  <see cref="Resume"/> builds its network from the
    /// document, so it owns that network outright.
    /// </para>
    /// </remarks>
    /// <param name="network">
    /// The network to train, which the session trains in place, and whose optimizer state it discards.
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
    /// <remarks>
    /// A session saved before the cost function replaced the cost derivative is refused:  its settings name a
    /// <c>CostDerivative</c>, which this library no longer reads.  The weights are not lost --
    /// <see cref="ReadNetwork"/> takes the network out of such a document, and a session can be started from it.
    /// </remarks>
    /// <param name="json">
    /// JSON holding a saved session.
    /// </param>
    /// <returns>
    /// Returns the session, carrying the network, settings, optimizer state and epoch count it was saved with.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// The JSON holds no session, or holds one this library cannot read.
    /// </exception>
    public static TrainingSession Resume(string json)
    {
        json.ThrowIfNullOrEmptyOrWhitespace();

        RefuseSessionSavedBeforeTheCostFunction(json);

        return JsonSerializer.Deserialize<TrainingSession>(json)
            ?? throw new ArgumentException("The JSON holds no training session.", nameof(json));
    }

    /// <summary>
    /// Reads the network out of a saved session, without the training that was going on around it.
    /// </summary>
    /// <remarks>
    /// What a caller is left with when a session cannot be resumed -- its settings naming a cost this library no
    /// longer reads -- since the network's own document has not changed and the weights are the part worth
    /// keeping.  A session started from the network returned here has an optimizer that has taken no steps.
    /// </remarks>
    /// <param name="json">
    /// JSON holding a saved session.
    /// </param>
    /// <returns>
    /// Returns the network the session was training.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// The JSON holds no session, or the session holds no network this library can read.
    /// </exception>
    public static INeuralNetwork ReadNetwork(string json)
    {
        json.ThrowIfNullOrEmptyOrWhitespace();

        JsonNode node = JsonNode.Parse(json)
            ?? throw new ArgumentException("The JSON holds no training session.", nameof(json));

        JsonNode network = node[nameof(Network)]
            ?? throw new ArgumentException(
                $"The saved session carries no '{nameof(Network)}'.",
                nameof(json));

        return network.Deserialize<INeuralNetwork>()
            ?? throw new ArgumentException("The saved session's network is null.", nameof(json));
    }

    #endregion

    #region Private Static

    /// <summary>
    /// Refuses a session document written before the cost function replaced the cost derivative.
    /// </summary>
    /// <remarks>
    /// Such a document names its cost <c>CostDerivative</c>, which nothing reads now, so the serializer would
    /// leave the settings without a cost and the failure would name a constructor parameter the caller never
    /// supplied.  Saying what the document is costs one pass over its properties, on the resume path only.
    /// </remarks>
    /// <param name="json">
    /// JSON holding a saved session.
    /// </param>
    /// <exception cref="ArgumentException">
    /// The document names a cost derivative rather than a cost function.
    /// </exception>
    private static void RefuseSessionSavedBeforeTheCostFunction(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);

        if (!document.RootElement.TryGetProperty(nameof(Settings), out JsonElement settings)
            || settings.ValueKind != JsonValueKind.Object
            || !settings.TryGetProperty("CostDerivative", out JsonElement _))
        {
            return;
        }

        throw new ArgumentException(
            "This session was saved before the library's cost derivative became a cost function, so its "
            + $"settings name a 'CostDerivative' where a '{nameof(TrainingSettings.CostFunction)}' is now read.  "
            + $"Take the network out with {nameof(ReadNetwork)} and start a session from it under new settings.",
            nameof(json));
    }

    /// <summary>
    /// Returns a network as the type the library builds, which is the only kind a session can train.
    /// </summary>
    /// <remarks>
    /// Every way of starting a session comes through here, so a network from elsewhere is refused by one
    /// message however it arrives.  It is asked before anything else happens to the network, since a copy
    /// through the serializer would refuse an unregistered type first and say so in its own terms.
    /// </remarks>
    /// <param name="network">
    /// The network a caller supplied.
    /// </param>
    /// <returns>
    /// Returns that network, as the library's own type.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// The network is not one this library built.
    /// </exception>
    private static NeuralNetwork AsTrainedNetwork(INeuralNetwork network)
    {
        return network as NeuralNetwork
            ?? throw new ArgumentException(
                $"A training session trains a network this library built, and '{network.GetType().Name}' "
                + "is not one.",
                nameof(network));
    }

    /// <summary>
    /// Copies a network, through the JSON it serializes to.
    /// </summary>
    /// <param name="network">
    /// The network to copy.
    /// </param>
    /// <returns>
    /// Returns a network holding the same weights, sharing nothing with the one it was copied from.
    /// </returns>
    private static NeuralNetwork Copy(NeuralNetwork network)
    {
        INeuralNetwork? copy = JsonSerializer.Deserialize<INeuralNetwork>(
            JsonSerializer.Serialize<INeuralNetwork>(network));

        return AsTrainedNetwork(
            copy ?? throw new ArgumentException("The network serialized to nothing.", nameof(network)));
    }

    /// <summary>
    /// Takes a batch's worth of samples from a set of them.
    /// </summary>
    /// <remarks>
    /// Read by index rather than skipped over, so that an epoch costs one pass over its samples whatever kind
    /// of list they arrived in.
    /// </remarks>
    /// <param name="samples">
    /// Every sample of the epoch.
    /// </param>
    /// <param name="firstIndex">
    /// Index of the batch's first sample.
    /// </param>
    /// <param name="count">
    /// Number of samples in the batch.
    /// </param>
    /// <returns>
    /// Returns the batch's samples, in the order they were given.
    /// </returns>
    private static IReadOnlyList<IReadOnlyList<float>> Slice(
        IReadOnlyList<IReadOnlyList<float>> samples,
        int firstIndex,
        int count)
    {
        IReadOnlyList<float>[] batch = new IReadOnlyList<float>[count];

        for (int index = 0; index < count; index++)
            batch[index] = samples[firstIndex + index];

        return batch;
    }

    #endregion

    #region Construction

    /// <summary>
    /// Constructs a session, which a caller does through <see cref="Start"/>, <see cref="Adopt"/> or
    /// <see cref="Resume"/>.
    /// </summary>
    /// <remarks>
    /// Not public:  those three are the ways to have a session, and each says what it does with the network it
    /// is given.  The serializer reaches this when a saved session is resumed.
    /// </remarks>
    /// <param name="network">
    /// The network the session trains, in place.
    /// </param>
    /// <param name="settings">
    /// Settings governing the training.
    /// </param>
    /// <param name="completedEpochCount">
    /// Number of epochs already completed.
    /// </param>
    /// <param name="optimizerState">
    /// State for the optimizer to continue from, one entry for each of the network's parameter groups, or
    /// <see langword="null"/> for an optimizer that starts afresh -- which also discards any state the network
    /// is still carrying from an earlier session.
    /// </param>
    /// <exception cref="ArgumentException">
    /// The network is not one this library built, or the optimizer state does not describe its parameters.
    /// </exception>
    [JsonConstructor]
    private TrainingSession(
        INeuralNetwork network,
        TrainingSettings settings,
        int completedEpochCount,
        IReadOnlyList<IOptimizerState?>? optimizerState)
    {
        network.ThrowIfNull();
        settings.ThrowIfNull();
        completedEpochCount.ThrowIfNegative();

        TrainedNetwork = AsTrainedNetwork(network);
        Settings = settings;
        CompletedEpochCount = completedEpochCount;

        if (optimizerState is null)
        {
            // An adopted network may have been trained by an earlier session, whose optimizer state its layers
            //  are still holding.  This session's optimizer has taken no steps, so it starts from none.
            TrainedNetwork.DiscardOptimizerState();
        }
        else
        {
            RestoreOptimizerState(optimizerState);
        }
    }

    #endregion

    #region Public

    /// <summary>
    /// Describes the session:  the network it trains, its optimizer, and how far it has gone.
    /// </summary>
    /// <returns>
    /// Returns the network's short description, the optimizer's short name, and the completed epoch count.
    /// </returns>
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
    /// session whose optimizer keeps no state, or which has taken no step, carries nothing.  Not public:  the
    /// order the entries are in is the network's business, and a caller has nothing it can do with them.
    /// </remarks>
    [JsonInclude]
    private IReadOnlyList<IOptimizerState?>? OptimizerState
    {
        get
        {
            IReadOnlyList<IOptimizerState?> state = TrainedNetwork.ExportOptimizerState();

            return state.Any(entry => entry is not null) ? state : null;
        }
    }

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
    /// Returns the sample's cost, measured against the parameters as they were before they moved.
    /// </returns>
    public async Task<float> TrainSampleAsync(
        IReadOnlyList<float> input,
        IReadOnlyList<float> targetOutput,
        CancellationToken cancellationToken = default)
    {
        input.ThrowIfNull();
        targetOutput.ThrowIfNull();

        return await TrainBatchAsync([input], [targetOutput], cancellationToken);
    }

    /// <summary>
    /// Trains the network on a batch of samples, moving its parameters once.
    /// </summary>
    /// <remarks>
    /// Every sample in the batch is learned from before any parameter moves, and each parameter then moves
    /// against the gradients the samples called for, averaged over the batch.  A batch is the samples it is
    /// given, so a batch of one is <see cref="TrainSampleAsync"/>.
    /// <para>
    /// A batch that is abandoned before its parameters begin to move -- cancelled, or refused for a gradient no
    /// limit can scale -- leaves the network exactly as it found it, gradients included, and the session can be
    /// trained again.  Once the parameters have begun to move, only an optimizer of the caller's own can fail,
    /// since every optimizer this library provides has had its state settled before the first group moves;  such
    /// a failure may leave some groups moved and others not, and the session cannot tell which, so what the
    /// network holds afterwards is neither what the batch started from nor what it would have produced.  A
    /// caller whose own optimizer can fail that way owes itself a copy of the network taken beforehand.
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
    /// Returns the batch's cost:  the mean over its samples, measured against the parameters as they were
    /// before they moved, which is what a caller watches to see whether training is working.
    /// </returns>
    public async Task<float> TrainBatchAsync(
        IReadOnlyList<IReadOnlyList<float>> input,
        IReadOnlyList<IReadOnlyList<float>> targetOutput,
        CancellationToken cancellationToken = default)
    {
        input.ThrowIfNull();
        targetOutput.ThrowIfNull();

        return await TrainBatchAsync(
            ReadLearningRate(),
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
    /// Returns the epoch's cost:  the mean over its samples of what each cost when it was learned from, so a
    /// caller training epoch after epoch can watch the figure fall.
    /// </returns>
    public async Task<float> TrainEpochAsync(
        IReadOnlyList<IReadOnlyList<float>> input,
        IReadOnlyList<IReadOnlyList<float>> targetOutput,
        int batchSize,
        CancellationToken cancellationToken = default)
    {
        input.ThrowIfNull();
        input.ThrowIfEmpty();
        targetOutput.ThrowIfNull();
        targetOutput.Count.ThrowIfNotEqualTo(input.Count);
        batchSize.ThrowIfNotPositive();

        float learningRate = ReadLearningRate();

        double costOfEverySample = 0;

        for (int firstIndex = 0; firstIndex < input.Count; firstIndex += batchSize)
        {
            int sampleCount = Math.Min(batchSize, input.Count - firstIndex);

            float batchCost = await TrainBatchAsync(
                learningRate,
                Slice(input, firstIndex, sampleCount),
                Slice(targetOutput, firstIndex, sampleCount),
                cancellationToken);

            // Weighted by the batch's size, so that a final partial batch does not count for as much as a full
            //  one:  the epoch's cost is the mean over its samples rather than over its batches.
            costOfEverySample += (double)batchCost * sampleCount;
        }

        CompletedEpochCount += 1;

        return (float)(costOfEverySample / input.Count);
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

    /// <summary>
    /// Reads the rate for the epoch in progress from the schedule, refusing one that would train nothing.
    /// </summary>
    /// <remarks>
    /// Checked here rather than only where a schedule is constructed, because a schedule is free to return a
    /// rate it was never constructed with:  an exponential decay underflows to zero after enough epochs, and a
    /// rate of zero would run every batch, count every epoch, advance the optimizer's state, and move no
    /// parameter.  The refusal names the schedule and the epoch, since the rate is the schedule's answer rather
    /// than anything the caller passed in.
    /// </remarks>
    /// <returns>
    /// Returns the learning rate for the epoch in progress.
    /// </returns>
    /// <exception cref="InvalidOperationException">
    /// The schedule returned a rate that is not positive.
    /// </exception>
    private float ReadLearningRate()
    {
        float learningRate = Settings.LearningRateScheduler.GetLearningRate(CompletedEpochCount);

        if (learningRate > 0)
            return learningRate;

        throw new InvalidOperationException(
            $"{Settings.LearningRateScheduler.GetType().Name} returned a learning rate of {learningRate} for "
            + $"epoch {CompletedEpochCount}, and a rate that is not positive would train every batch without "
            + "moving a parameter.");
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
        IReadOnlyList<int> parameterCounts = TrainedNetwork.GetParameterCounts();

        if (optimizerState.Count != parameterCounts.Count)
        {
            throw new ArgumentException(
                $"The session carries optimizer state for {optimizerState.Count} parameter groups, and its "
                + $"network has {parameterCounts.Count}.",
                nameof(optimizerState));
        }

        // Checked group by group, and against this session's own optimizer:  state of the right count and the
        //  wrong shape, or state another optimizer created, would otherwise be imported here and then quietly
        //  replaced at the first step, which is the discard this refusal exists to prevent.
        for (int groupIndex = 0; groupIndex < optimizerState.Count; groupIndex++)
        {
            IOptimizerState? state = optimizerState[groupIndex];

            if (Settings.Optimizer.CanApplyTo(state, parameterCounts[groupIndex]))
                continue;

            throw new ArgumentException(
                $"The optimizer state saved for parameter group {groupIndex} is "
                + $"'{state?.GetType().Name ?? "null"}' for {state?.ParameterCount ?? 0} parameters, and "
                + $"{Settings.Optimizer.ShortName} cannot continue from it for the "
                + $"{parameterCounts[groupIndex]} parameters that group holds.",
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
    /// Returns the mean cost of the batch's samples, measured as each was learned from.
    /// </returns>
    private async Task<float> TrainBatchAsync(
        float learningRate,
        IReadOnlyList<IReadOnlyList<float>> input,
        IReadOnlyList<IReadOnlyList<float>> targetOutput,
        CancellationToken cancellationToken)
    {
        input.ThrowIfEmpty();
        targetOutput.Count.ThrowIfNotEqualTo(input.Count);

        double costOfEverySample = 0;

        try
        {
            for (int sampleIndex = 0; sampleIndex < input.Count; sampleIndex++)
            {
                costOfEverySample += await TrainedNetwork.LearnFromSampleAsync(
                    Settings.CostFunction,
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

        try
        {
            TrainedNetwork.ApplyGradients(Settings.Optimizer, learningRate, input.Count, Settings.MaxGradientNorm);
        }
        catch
        {
            // Separate from the accumulation above because the two failures leave different things behind:
            //  nothing has moved there, and here the parameters move group by group, so a failure inside the
            //  step may leave some groups moved and others not.  The gradients go either way.
            TrainedNetwork.DiscardGradients();

            throw;
        }

        return (float)(costOfEverySample / input.Count);
    }

    #endregion
}
