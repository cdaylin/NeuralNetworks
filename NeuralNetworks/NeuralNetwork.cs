using Daylin.NeuralNetworks.Backpropagation;
using Daylin.NeuralNetworks.Backpropagation.Optimization;
using Daylin.NeuralNetworks.Layers;
using Daylin.NeuralNetworks.Layers.CompositeLayers;
using Daylin.NeuralNetworks.Layers.NeuralLayers;

using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks;

internal class NeuralNetwork : INeuralNetwork
{
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
    /// Number of groups the network's parameters are kept in, which is the shape of its training state.
    /// </summary>
    /// <remarks>
    /// A <see cref="TrainingSession"/> compares this against the state it holds, so that state built for one
    /// network is not imported into a differently shaped one.
    /// </remarks>
    [JsonIgnore]
    internal int ParameterGroupCount => ParameterGroups.Count();

    /// <summary>
    /// Accumulates the gradients one training sample calls for, without moving any parameter.
    /// </summary>
    /// <param name="costDerivative">
    /// The function calculating the derivative of the cost with respect to each output value.
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
    /// Returns a task representing this <c>async</c> operation.
    /// </returns>
    internal async Task LearnFromSampleAsync(
        ICostDerivative costDerivative,
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

        IEnumerable<float> gradients = CalculateOutputGradients(
            costDerivative: costDerivative,
            targetOutput: targetOutput,
            actualOutput: output);

        cancellationToken.ThrowIfCancellationRequested();

        await backpropagateGradientsAsync(gradients: gradients, cancellationToken: cancellationToken);
    }

    /// <summary>
    /// Moves every parameter of the network by the gradients accumulated for it, and discards them.
    /// </summary>
    /// <remarks>
    /// Parameters are moved group by group, in the order the network's layers hold them, after the norm of
    /// the whole network's gradients has been taken.  A limit on that norm is what makes the network rather
    /// than any one layer the place parameters move:  no group can move until every group's gradients are
    /// known.
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
        IReadOnlyList<ParameterGroup> parameterGroups = [.. ParameterGroups];

        float gradientScale = CalculateGradientScale(parameterGroups, sampleCount, maxGradientNorm);

        foreach (ParameterGroup parameterGroup in parameterGroups)
            parameterGroup.ApplyGradients(optimizer, learningRate, sampleCount, gradientScale);
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
        optimizerState.Count.ThrowIfNotEqualTo(ParameterGroupCount);

        int groupIndex = 0;

        foreach (ParameterGroup parameterGroup in ParameterGroups)
            parameterGroup.OptimizerState = optimizerState[groupIndex++];
    }

    #endregion

    #region Private Static

    /// <summary>
    /// Calculates the factor that scales the network's mean gradients down to a maximum norm.
    /// </summary>
    /// <param name="parameterGroups">
    /// Every parameter group of the network.
    /// </param>
    /// <param name="sampleCount">
    /// Number of samples the gradients were accumulated from.
    /// </param>
    /// <param name="maxGradientNorm">
    /// The largest the norm may be, or <see langword="null"/> for no limit.
    /// </param>
    /// <returns>
    /// Returns the maximum over the norm where the norm exceeds the maximum, and one otherwise.
    /// </returns>
    private static float CalculateGradientScale(
        IReadOnlyList<ParameterGroup> parameterGroups,
        int sampleCount,
        float? maxGradientNorm)
    {
        if (maxGradientNorm is not float maximum)
            return 1;

        // Summed in the order the groups are held, which is fixed, so the norm is the same on every run.
        double norm = Math.Sqrt(parameterGroups.Sum(group => group.CalculateSquaredGradientNorm(sampleCount)));

        return norm > maximum ? (float)(maximum / norm) : 1;
    }

    /// <summary>
    /// Walks a layer tree, in a fixed order, for the parameter groups its neural layers hold.
    /// </summary>
    /// <param name="layer">
    /// The layer to walk.
    /// </param>
    /// <returns>
    /// Returns every parameter group the layer holds, directly or through the layers it contains.
    /// </returns>
    private static IEnumerable<ParameterGroup> FindParameterGroups(INetworkLayer layer)
    {
        return layer switch
        {
            NeuralLayer neuralLayer => neuralLayer.ParameterGroups,
            CompositeNetworkLayer compositeLayer => compositeLayer.Layers.SelectMany(FindParameterGroups),
            _ => [],
        };
    }

    #endregion

    #region Private

    [JsonInclude]
    private INetworkLayer NetworkLayer { get; }

    [JsonIgnore]
    private IEnumerable<ParameterGroup> ParameterGroups => FindParameterGroups(NetworkLayer);

    /// <summary>
    /// Calculates the partial derivative of the cost function with respect to each network output value.
    /// </summary>
    private IEnumerable<float> CalculateOutputGradients(
        ICostDerivative costDerivative,
        IEnumerable<float> targetOutput,
        IEnumerable<float> actualOutput)
    {
        return targetOutput
            .Zip(actualOutput)
            .Select(pair => costDerivative.Calculate(pair.First, pair.Second));
    }

    #endregion
}
