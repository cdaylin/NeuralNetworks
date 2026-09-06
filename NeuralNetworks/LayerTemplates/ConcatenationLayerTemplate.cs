using Daylin.NeuralNetworks.Layers;
using Daylin.NeuralNetworks.Layers.CompositeLayers;

namespace Daylin.NeuralNetworks.LayerTemplates;

/// <summary>
/// A template for constructing a composite network layer comprising two or more nested layers that are
/// concatenated to behave as a single layer.
/// </summary>
/// <remarks>
/// The input to a concatenation layer comprises the input to all nested layers concatenated in the order provided
/// to the constructor.  The output from a concatenation layer is the output from all nested layers concatenated
/// in the same order.
/// </remarks>
public class ConcatenationLayerTemplate : CompositeNetworkLayerTemplate
{
    public ConcatenationLayerTemplate(IEnumerable<INetworkLayerTemplate> layers)
        : base(Unwrap(layers))
    {
    }

    public override int InputCount => LayerTemplates.Select(layer => layer.InputCount).Sum();

    public override int OutputCount => LayerTemplates.Select(layer => layer.OutputCount).Sum();

    public override INetworkLayer Build()
    {
        return new ConcatenationLayer(
            LayerTemplates.Select(template => template.Build()));
    }

    private static IEnumerable<INetworkLayerTemplate> Unwrap(
        IEnumerable<INetworkLayerTemplate>? layerDefinitions)
    {
        if (layerDefinitions is null)
            return Enumerable.Empty<INetworkLayerTemplate>();

        return layerDefinitions.SelectMany(Unwrap);
    }

    private static IEnumerable<INetworkLayerTemplate> Unwrap(INetworkLayerTemplate layerDefinition)
    {
        if (layerDefinition is ConcatenationLayerTemplate concatenationDefinition)
            return concatenationDefinition.LayerTemplates;

        return layerDefinition.ToEnumerable();
    }
}
