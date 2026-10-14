using Daylin.NeuralNetworks.Backpropagation;
using Daylin.NeuralNetworks.Backpropagation.Cost;
using Daylin.NeuralNetworks.Backpropagation.Optimization;
using Daylin.NeuralNetworks.Layers;
using Daylin.NeuralNetworks.Layers.CompositeLayers;
using Daylin.NeuralNetworks.Layers.NeuralLayers;

using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks;

internal class NeuralNetwork : INeuralNetwork
{
    #region Private Static

    /// <summary>
    /// Calculates the factor that scales the network's mean gradients down to a maximum norm.
    /// </summary>
    /// <param name="norm">
    /// The norm of the network's mean gradients:  the square root of the sum of their squares.
    /// </param>
    /// <param name="maxGradientNorm">
    /// The largest the norm may be, or <see langword="null"/> for no limit.
    /// </param>
    /// <returns>
    /// Returns the maximum over the norm where the norm exceeds the maximum, and one otherwise.  The factor is a
    /// <see langword="double"/>, since a large norm against a small maximum gives one too small for a
    /// <see langword="float"/> to hold.
    /// </returns>
    /// <exception cref="InvalidOperationException">
    /// The norm is not a finite number, which no factor can scale to the maximum.
    /// </exception>
    private static double CalculateGradientScale(double norm, float? maxGradientNorm)
    {
        if (maxGradientNorm is not float maximum)
            return 1;

        // A gradient that has overflowed to infinity, or a sample that carried a value that is not a number,
        //  reaches here as a norm no factor can bring within the maximum.  Without this refusal a norm that is
        //  not a number compares false against the maximum however the comparison is written, so the limit would
        //  silently not apply;  and an infinite norm gives a factor of zero, which multiplied into the infinite
        //  gradient itself yields a value that is not a number rather than no movement.  Refusing says which of
        //  the two it was.
        if (!double.IsFinite(norm))
        {
            throw new InvalidOperationException(
                $"The norm of the network's gradients is {norm}, so the maximum of {maximum} cannot be applied "
                + "to them.  A gradient that is not a finite number comes from a sample, a target value, or a "
                + "learning rate that drove the network's parameters past what a float can hold.");
        }

        return norm > maximum ? maximum / norm : 1;
    }

    #endregion

    #region Construction

    [JsonConstructor]
    public NeuralNetwork(INetworkLayer networkLayer)
    {
        networkLayer.ThrowIfNull();

        NetworkLayer = networkLayer;
    }

    #endregion

    #region Public

    public override string ToString() => ShortDescription;

    [JsonIgnore]
    public string ShortDescription => NetworkLayer.ShortDescription;

    [JsonIgnore]
    public int InputCount => NetworkLayer.InputCount;

    [JsonIgnore]
    public int OutputCount => NetworkLayer.OutputCount;

    public void Initialize() => NetworkLayer.Initialize(Enumerable.Repeat(1, OutputCount));

    public async Task<IEnumerable<float>> CalculateOutputAsync(
        IEnumerable<float> input,
        CancellationToken cancellationToken = default)
    {
        return await NetworkLayer.CalculateOutputAsync(input, cancellationToken);
    }

    #endregion

    #region Internal

    /// <summary>
    /// Accumulates the gradients one training sample calls for, without moving any parameter.
    /// </summary>
    /// <param name="costFunction">
    /// The cost being minimized, which the sample's cost is measured with and whose derivative the gradients
    /// are calculated from.
    /// </param>
    /// <param name="input">
    /// Input values.
    /// </param>
    /// <param name="targetOutput">
    /// Desired output values.
    /// </param>
    /// <param name="cancellationToken">
    /// A cancellation token.
    /// </param>
    /// <returns>
    /// Returns the cost of the sample:  the mean over the network's output values of what the cost function
    /// made of each, measured against the parameters as they were before the sample was learned from.
    /// </returns>
    internal async Task<float> LearnFromSampleAsync(
        ICostFunction costFunction,
        IReadOnlyList<float> input,
        IReadOnlyList<float> targetOutput,
        CancellationToken cancellationToken)
    {
        input.Count.ThrowIfNotEqualTo(NetworkLayer.InputCount);
        targetOutput.Count.ThrowIfNotEqualTo(NetworkLayer.OutputCount);

        (IEnumerable<float> output, BackpropagateGradientsAsync backpropagateGradientsAsync) =
            await NetworkLayer.FeedForwardAsync(
                input: input,
                backpropagateGradientsAsync: null,
                cancellationToken: cancellationToken);

        cancellationToken.ThrowIfCancellationRequested();

        IReadOnlyList<float> actualOutput = [.. output];

        float cost = CalculateCost(costFunction, targetOutput, actualOutput);

        IEnumerable<float> gradients = CalculateOutputGradients(
            costFunction: costFunction,
            targetOutput: targetOutput,
            actualOutput: actualOutput);

        cancellationToken.ThrowIfCancellationRequested();

        await backpropagateGradientsAsync(gradients: gradients, cancellationToken: cancellationToken);

        return cost;
    }

    /// <summary>
    /// Moves every parameter of the network by the gradients accumulated for it, and discards them.
    /// </summary>
    /// <remarks>
    /// Three passes over the groups:  each takes the mean of its gradients and contributes to the norm of the
    /// whole network's, then each is given the optimizer state its step needs, then each moves.  A limit on the
    /// norm is what makes the network rather than any one layer the place parameters move, since no group can
    /// move until every group's gradients are known;  settling the state in a pass of its own is what keeps a
    /// failure to allocate it from stopping halfway through a network that has already begun to move.
    /// <para>
    /// Only the third pass moves anything, which is what lets a caller tell a batch that was refused from one
    /// that was abandoned partway.
    /// </para>
    /// </remarks>
    /// <param name="optimizer">
    /// The optimizer moving the parameters.
    /// </param>
    /// <param name="learningRate">
    /// The learning rate for the training epoch in progress.
    /// </param>
    /// <param name="sampleCount">
    /// Number of samples the gradients were accumulated from, which they are averaged over.
    /// </param>
    /// <param name="maxGradientNorm">
    /// The largest the norm of the mean gradients may be before they are scaled down to it, or
    /// <see langword="null"/> for them never to be scaled.
    /// </param>
    internal void ApplyGradients(IOptimizer optimizer, float learningRate, int sampleCount, float? maxGradientNorm)
    {
        optimizer.ThrowIfNull();
        sampleCount.ThrowIfNotPositive();

        bool measureNorm = maxGradientNorm is not null;

        double squaredNorm = 0;

        foreach (ParameterGroup parameterGroup in ParameterGroups)
            squaredNorm += parameterGroup.TakeMeanGradients(sampleCount, measureNorm);

        double gradientScale = CalculateGradientScale(Math.Sqrt(squaredNorm), maxGradientNorm);

        // Every group is given the state it needs before any parameter moves, since creating state allocates and
        //  a failure once the moving has started would leave some groups stepped and others not.
        foreach (ParameterGroup parameterGroup in ParameterGroups)
            parameterGroup.PrepareOptimizerState(optimizer);

        foreach (ParameterGroup parameterGroup in ParameterGroups)
            parameterGroup.ApplyGradients(optimizer, learningRate, gradientScale);
    }

    /// <summary>
    /// Discards the optimizer state the network's layers are holding, leaving the parameters alone.
    /// </summary>
    /// <remarks>
    /// This is what lets a session adopt a network another session trained and begin with an optimizer that has
    /// taken no steps.
    /// </remarks>
    internal void DiscardOptimizerState()
    {
        foreach (ParameterGroup parameterGroup in ParameterGroups)
            parameterGroup.OptimizerState = null;
    }

    /// <summary>
    /// Discards the gradients accumulated for every parameter of the network, leaving the parameters alone.
    /// </summary>
    /// <remarks>
    /// This is what leaves a network untouched by a batch that was abandoned partway through.
    /// </remarks>
    internal void DiscardGradients()
    {
        foreach (ParameterGroup parameterGroup in ParameterGroups)
            parameterGroup.DiscardGradients();
    }

    /// <summary>
    /// Takes the optimizer state the network's layers are holding, in a fixed order.
    /// </summary>
    /// <returns>
    /// Returns one entry for each parameter group, which is <see langword="null"/> where no optimizer has
    /// reached that group or the optimizer that did keeps no state.
    /// </returns>
    internal IReadOnlyList<IOptimizerState?> ExportOptimizerState() =>
        ParameterGroups.Select(parameterGroup => parameterGroup.OptimizerState).ToList();

    /// <summary>
    /// Gives the network's layers the optimizer state to continue from.
    /// </summary>
    /// <param name="optimizerState">
    /// One entry for each parameter group, in the order <see cref="ExportOptimizerState"/> returns them.
    /// </param>
    internal void ImportOptimizerState(IReadOnlyList<IOptimizerState?> optimizerState)
    {
        optimizerState.ThrowIfNull();
        optimizerState.Count.ThrowIfNotEqualTo(ParameterGroups.Count);

        for (int groupIndex = 0; groupIndex < ParameterGroups.Count; groupIndex++)
            ParameterGroups[groupIndex].OptimizerState = optimizerState[groupIndex];
    }

    /// <summary>
    /// Number of parameters in each of the network's groups, in the order the groups are held.
    /// </summary>
    /// <remarks>
    /// A <see cref="TrainingSession"/> compares optimizer state against this before importing it, so that
    /// state built for one shape of network is refused rather than silently replaced at the first step.
    /// </remarks>
    /// <returns>
    /// Returns one count for each parameter group.
    /// </returns>
    internal IReadOnlyList<int> GetParameterCounts() =>
        [.. ParameterGroups.Select(parameterGroup => parameterGroup.Parameters.Length)];

    #endregion

    #region Private

    [JsonInclude]
    private INetworkLayer NetworkLayer { get; }

    /// <summary>
    /// Every parameter group the network's layers hold, in the order they hold them.
    /// </summary>
    /// <remarks>
    /// Found once and kept:  a layer tree does not change after it is built, and each layer holds its own
    /// groups for its lifetime, so the list and the order are stable.  Optimizer state is exported and imported
    /// under that order.  A layer that holds no parameters, and one that is not the library's own, contribute
    /// none.
    /// </remarks>
    [JsonIgnore]
    private IReadOnlyList<ParameterGroup> ParameterGroups =>
        field ??= NetworkLayer is IParameterizedLayer parameterizedLayer ? parameterizedLayer.ParameterGroups : [];

    /// <summary>
    /// Calculates the partial derivative of the cost function with respect to each network output value.
    /// </summary>
    private static IEnumerable<float> CalculateOutputGradients(
        ICostFunction costFunction,
        IEnumerable<float> targetOutput,
        IEnumerable<float> actualOutput)
    {
        return targetOutput
            .Zip(actualOutput)
            .Select(pair => costFunction.CalculateDerivative(pair.First, pair.Second));
    }

    /// <summary>
    /// Calculates what the cost function makes of a sample's output, as a single number.
    /// </summary>
    /// <remarks>
    /// The mean over the output values rather than their sum, so that the figure a caller watches means the same
    /// thing whatever the network's output count is.
    /// </remarks>
    /// <param name="costFunction">
    /// The cost being minimized.
    /// </param>
    /// <param name="targetOutput">
    /// Desired output values.
    /// </param>
    /// <param name="actualOutput">
    /// The output values the network calculated.
    /// </param>
    /// <returns>
    /// Returns the sample's cost.
    /// </returns>
    private static float CalculateCost(
        ICostFunction costFunction,
        IReadOnlyList<float> targetOutput,
        IReadOnlyList<float> actualOutput)
    {
        double sum = 0;

        for (int index = 0; index < targetOutput.Count; index++)
            sum += costFunction.Calculate(targetOutput[index], actualOutput[index]);

        return (float)(sum / targetOutput.Count);
    }

    #endregion
}
