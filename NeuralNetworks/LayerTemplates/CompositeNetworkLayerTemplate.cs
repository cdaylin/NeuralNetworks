using Daylin.NeuralNetworks.Layers;

namespace Daylin.NeuralNetworks.LayerTemplates;

/// <summary>
/// A template for constructing a composite network layer comprising two or more nested layers.
/// </summary>
public abstract class CompositeNetworkLayerTemplate : INetworkLayerTemplate
{
    protected CompositeNetworkLayerTemplate(IEnumerable<INetworkLayerTemplate> layerTemplates)
    {
        layerTemplates.ThrowIfNull();
        layerTemplates.ThrowIfContainsNull();
        layerTemplates.Count().ThrowIfLessThan(2);

        LayerTemplates = layerTemplates;
    }

    public IEnumerable<INetworkLayerTemplate> LayerTemplates
    {
        get => layerTemplates!;
        private set => layerTemplates = value.ToArray();
    }

    private INetworkLayerTemplate[]? layerTemplates;

    public abstract int InputCount { get; }

    public abstract int OutputCount { get; }

    public abstract INetworkLayer Build();
}
