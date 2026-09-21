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

    public override async Task<(IEnumerable<float>, BackpropagateGradientsAsync)> FeedForwardAsync(
        IEnumerable<float> input,
        BackpropagateGradientsAsync? backpropagateGradientsAsync,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        // create a handler for gathering results from contained layers
        GradientBackpropagationHandler? backpropagationHandler = backpropagateGradientsAsync is null
            ? null
            : new GradientBackpropagationHandler(this, backpropagateGradientsAsync);

        // feed-forward through contained layers
        List<(IEnumerable<float>, BackpropagateGradientsAsync)> feedForwardResults =
            await FeedForwardThroughContainedLayersAsync(input, backpropagationHandler, cancellationToken).ToListAsync();

        // reassemble output from contained layers
        IEnumerable<float> output = feedForwardResults.SelectMany(result => result.Item1);

        return (output, BackpropagateGradientsAsync);

        async Task BackpropagateGradientsAsync(IEnumerable<float> gradients, CancellationToken cancellationToken)
        {
            await BackpropagateGradientsToContainedLayersAsync(
                gradients,
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

    private async IAsyncEnumerable<(IEnumerable<float>, BackpropagateGradientsAsync)>
        FeedForwardThroughContainedLayersAsync(
            IEnumerable<float> input,
            GradientBackpropagationHandler? backpropagationHandler,
            [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        int skipCount = 0;

        foreach (INetworkLayer layer in Layers)
        {
            cancellationToken.ThrowIfCancellationRequested();

            BackpropagateGradientsAsync? backpropagateAsync = backpropagationHandler is null
                ? null
                : (gradients, cancellationToken) => backpropagationHandler.GatherGradientsAsync(
                    layer, gradients, cancellationToken);

            IEnumerable<float> layerInput = input.Skip(skipCount).Take(layer.InputCount);

            yield return await layer.FeedForwardAsync(
                layerInput,
                backpropagateAsync,
                cancellationToken);

            skipCount += layer.InputCount;
        }
    }

    private async Task BackpropagateGradientsToContainedLayersAsync(
        IEnumerable<float> gradients,
        IEnumerable<BackpropagateGradientsAsync> backpropagationDelegates,
        CancellationToken cancellationToken)
    {
        int index = 0;
        int skipCount = 0;

        foreach (BackpropagateGradientsAsync backpropagateGradientsAsync in backpropagationDelegates)
        {
            INetworkLayer layer = Layers.ElementAt(index);

            IEnumerable<float> layerGradients = gradients.Skip(skipCount).Take(layer.OutputCount);

            await backpropagateGradientsAsync(layerGradients, cancellationToken);

            skipCount += layer.OutputCount;
            index += 1;
        }
    }

    /// <summary>
    /// Responsible for gathering gradients backpropagated from all contained layers and then backpropagating
    /// them to the preceeding layer.
    /// </summary>
    private class GradientBackpropagationHandler
    {
        public GradientBackpropagationHandler(
            ConcatenationLayer concatenationLayer,
            BackpropagateGradientsAsync backpropagateGradientsAsync)
        {
            ConcatenationLayer = concatenationLayer;
            BackpropagateGradientsAsync = backpropagateGradientsAsync;

            GradientsArray = new IEnumerable<float>[ConcatenationLayer.Layers.Count()];
        }

        public async Task GatherGradientsAsync(
            INetworkLayer layer,
            IEnumerable<float> gradients,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int layerIndex = ConcatenationLayer.Layers.IndexOf(layer);

            GradientsArray[layerIndex] = gradients;

            // if all gradients have been gathered, backpropagate them to preceeding layer
            if (Interlocked.Increment(ref completedLayerCount) == ConcatenationLayer.Layers.Count())
            {
                IEnumerable<float> combinedGradients =
                    GradientsArray.SelectMany(layerGradients => layerGradients);

                await BackpropagateGradientsAsync(combinedGradients, cancellationToken);
            }
        }

        private ConcatenationLayer ConcatenationLayer { get; }

        private BackpropagateGradientsAsync BackpropagateGradientsAsync { get; }

        private IEnumerable<float>[] GradientsArray { get; }

        private volatile int completedLayerCount;
    }
}