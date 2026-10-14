using Daylin.NeuralNetworks.Layers.NeuralLayers;

namespace Daylin.NeuralNetworks.Layers.CompositeLayers;

/// <summary>
/// A composite network component comprising two or more nested components.
/// </summary>
internal abstract class CompositeNetworkLayer : INetworkLayer, IParameterizedLayer
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

    /// <inheritdoc cref="IParameterizedLayer.ParameterGroups"/>
    /// <remarks>
    /// The nested layers' groups, in the order the layers are held, so that a composite presents its
    /// parameters exactly as the layers inside it hold them.  Implemented explicitly, so that the serializer
    /// does not see it:  a saved network carries no training state.
    /// </remarks>
    IReadOnlyList<ParameterGroup> IParameterizedLayer.ParameterGroups =>
        field ??= [.. Layers.OfType<IParameterizedLayer>().SelectMany(layer => layer.ParameterGroups)];

    public abstract int InputCount { get; }

    public abstract int OutputCount { get; }

    public abstract IEnumerable<int> Initialize(IEnumerable<int> outgoingConnectionCounts);

    public abstract Task<IEnumerable<float>> CalculateOutputAsync(
        IEnumerable<float> input,
        CancellationToken cancellationToken = default);

    public abstract Task<(IEnumerable<float>, BackpropagateGradientsAsync)> FeedForwardAsync(
        IEnumerable<float> input,
        BackpropagateGradientsAsync? backpropagateGradientsAsync,
        CancellationToken cancellationToken = default);
}
