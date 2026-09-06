namespace Daylin.NeuralNetworks.Layers.CompositeLayers;

/// <summary>
/// A composite network component comprising two or more nested components.
/// </summary>
internal abstract class CompositeNetworkLayer : INetworkLayer
{
    protected CompositeNetworkLayer(IEnumerable<INetworkLayer> layers)
    {
        layers.ThrowIfNull();
        layers.ThrowIfContainsNull();
        layers.Count().ThrowIfLessThan(2);

        Layers = layers.ToArray();
    }

    public abstract string ShortDescription { get; }

    public IEnumerable<INetworkLayer> Layers { get; }

    public abstract int InputCount { get; }

    public abstract int OutputCount { get; }

    public abstract IEnumerable<int> Initialize(IEnumerable<int> outgoingConnectionCounts);

    public abstract Task<IEnumerable<float>> CalculateOutputAsync(
        IEnumerable<float> input,
        CancellationToken cancellationToken = default);

    public abstract Task<(IEnumerable<float>, BackpropagateErrorsAsync)> FeedForwardAsync(
        IEnumerable<float> input,
        BackpropagateErrorsAsync? backpropagateErrorsAsync,
        CancellationToken cancellationToken = default);
}
