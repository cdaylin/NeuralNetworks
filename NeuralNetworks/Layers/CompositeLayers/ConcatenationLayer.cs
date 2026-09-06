using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Layers.CompositeLayers;

/// <summary>
/// A composite network layer comprising two or more nested layers that are concatenated to behave as a single
/// layer.
/// </summary>
/// <remarks>
/// The input to a concatenation layer comprises the input to all nested layers concatenated in the order provided
/// to the constructor.  The output from a concatenation layer is the output from all nested layers concatenated
/// in the same order.
/// </remarks>
internal sealed class ConcatenationLayer : CompositeNetworkLayer
{
    [JsonConstructor]
    public ConcatenationLayer(IEnumerable<INetworkLayer> layers)
        : base(layers)
    {
    }

    public override string ToString() => ShortDescription;

    [JsonIgnore]
    public override string ShortDescription
    {
        get
        {
            StringBuilder description = new();

            description.Append("(");
            description.Append(Layers.First().ShortDescription);

            foreach (INetworkLayer layer in Layers.Skip(1))
                description.Append($" + {layer.ShortDescription}");

            description.Append(")");

            return description.ToString();
        }
    }

    [JsonIgnore]
    public override int InputCount => Layers.Select(layer => layer.InputCount).Sum();

    [JsonIgnore]
    public override int OutputCount => Layers.Select(layer => layer.OutputCount).Sum();

    public override IEnumerable<int> Initialize(IEnumerable<int> outputConnectionCounts)
    {
        IEnumerable<int> inputConnectionCounts = Enumerable.Empty<int>();

        int skipCount = 0;

        foreach (INetworkLayer layer in Layers)
        {
            inputConnectionCounts = inputConnectionCounts.Concat(
                layer.Initialize(outputConnectionCounts.Skip(skipCount).Take(layer.OutputCount)));

            skipCount += layer.OutputCount;
        }

        return inputConnectionCounts;
    }

    public override async Task<IEnumerable<float>> CalculateOutputAsync(
        IEnumerable<float> input,
        CancellationToken cancellationToken = default)
    {
        return await CalculateOutputFromContainedLayersAsync(input, cancellationToken)
           .AggregateAsync((first, second) => first.Concat(second));
    }

    public override async Task<(IEnumerable<float>, BackpropagateErrorsAsync)> FeedForwardAsync(
        IEnumerable<float> input,
        BackpropagateErrorsAsync? backpropagateErrorsAsync,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        // create a handler for gathering results from contained layers
        ErrorBackpropagationHandler? backpropagationHandler = backpropagateErrorsAsync is null
            ? null
            : new ErrorBackpropagationHandler(this, backpropagateErrorsAsync);

        // feed-forward through contained layers
        List<(IEnumerable<float>, BackpropagateErrorsAsync)> feedForwardResults =
            await FeedForwardThroughContainedLayersAsync(input, backpropagationHandler, cancellationToken).ToListAsync();

        // reassemble output from contained layers
        IEnumerable<float> output = feedForwardResults.SelectMany(result => result.Item1);

        return (output, BackpropagateErrorsAsync);

        async Task BackpropagateErrorsAsync(
            float learningRate,
            IEnumerable<float> errors,
            CancellationToken cancellationToken)
        {
            await BackpropagateErrorsToContainedLayersAsync(
                learningRate,
                errors,
                feedForwardResults.Select(result => result.Item2),
                cancellationToken);
        }
    }

    private async IAsyncEnumerable<IEnumerable<float>> CalculateOutputFromContainedLayersAsync(
        IEnumerable<float> input,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        int skipCount = 0;

        foreach (INetworkLayer layer in Layers)
        {
            cancellationToken.ThrowIfCancellationRequested();

            IEnumerable<float> layerInput = input.Skip(skipCount).Take(layer.InputCount);

            yield return await layer.CalculateOutputAsync(layerInput, cancellationToken);

            skipCount += layer.InputCount;
        }
    }

    private async IAsyncEnumerable<(IEnumerable<float>, BackpropagateErrorsAsync)>
        FeedForwardThroughContainedLayersAsync(
            IEnumerable<float> input,
            ErrorBackpropagationHandler? backpropagationHandler,
            [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        int skipCount = 0;

        foreach (INetworkLayer layer in Layers)
        {
            cancellationToken.ThrowIfCancellationRequested();

            BackpropagateErrorsAsync? backpropagateAsync = backpropagationHandler is null
                ? null
                : (learningRate, errors, cancellationToken) => backpropagationHandler.GatherErrorsAsync(
                    layer, learningRate, errors, cancellationToken);

            IEnumerable<float> layerInput = input.Skip(skipCount).Take(layer.InputCount);

            yield return await layer.FeedForwardAsync(
                layerInput,
                backpropagateAsync,
                cancellationToken);

            skipCount += layer.InputCount;
        }
    }

    private async Task BackpropagateErrorsToContainedLayersAsync(
        float learningRate,
        IEnumerable<float> errors,
        IEnumerable<BackpropagateErrorsAsync> backpropagationDelegates,
        CancellationToken cancellationToken)
    {
        int index = 0;
        int skipCount = 0;

        foreach (BackpropagateErrorsAsync backpropagateErrorsAsync in backpropagationDelegates)
        {
            INetworkLayer layer = Layers.ElementAt(index);

            IEnumerable<float> layerErrors = errors.Skip(skipCount).Take(layer.OutputCount);

            await backpropagateErrorsAsync(
                learningRate,
                layerErrors,
                cancellationToken);

            skipCount += layer.OutputCount;
            index += 1;
        }
    }

    /// <summary>
    /// Responsible for gathering errors backpropagated from all contained layers and then backpropagating them
    /// to the preceeding layer.
    /// </summary>
    private class ErrorBackpropagationHandler
    {
        public ErrorBackpropagationHandler(
            ConcatenationLayer concatenationLayer,
            BackpropagateErrorsAsync backpropagateErrorsAsync)
        {
            ConcatenationLayer = concatenationLayer;
            BackpropagateErrorsAsync = backpropagateErrorsAsync;

            ErrorsArray = new IEnumerable<float>[ConcatenationLayer.Layers.Count()];
        }

        public async Task GatherErrorsAsync(
            INetworkLayer layer,
            float learningRate,
            IEnumerable<float> errors,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int layerIndex = ConcatenationLayer.Layers.IndexOf(layer);

            ErrorsArray[layerIndex] = errors;

            // if all errors have been gathered, backpropagate them to preceeding layer
            if (Interlocked.Increment(ref completedLayerCount) == ConcatenationLayer.Layers.Count())
            {
                IEnumerable<float> combinedErrors = ErrorsArray.SelectMany(errors => errors);

                await BackpropagateErrorsAsync(learningRate, combinedErrors, cancellationToken);
            }
        }

        private ConcatenationLayer ConcatenationLayer { get; }

        private BackpropagateErrorsAsync BackpropagateErrorsAsync { get; }

        private IEnumerable<float>[] ErrorsArray { get; }

        private volatile int completedLayerCount;
    }
}