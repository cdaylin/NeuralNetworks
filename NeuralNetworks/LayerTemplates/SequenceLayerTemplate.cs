using Daylin.NeuralNetworks.Layers;
using Daylin.NeuralNetworks.Layers.CompositeLayers;

namespace Daylin.NeuralNetworks.LayerTemplates;

/// <summary>
/// A template for constructing a composite network layer comprising two or more nested layers that are
/// connected sequentially.
/// </summary>
/// <remarks>
/// The input to a sequence layer is the input to the first nested layer.  The output from a sequence layer is the
/// output from the last nested layer.  Nested layers are connected sequentially, in the order provided to the
/// constructor.  The output from each nested layer is used as the input for the subsequent layer.  
/// </remarks>
public sealed class SequenceLayerTemplate : CompositeNetworkLayerTemplate
{
    public SequenceLayerTemplate(IEnumerable<INetworkLayerTemplate> layers)
        : base(Unwrap(layers))
    {
    }

    public override int InputCount => LayerTemplates.First().InputCount;

    public override int OutputCount => LayerTemplates.Last().OutputCount;

    public override INetworkLayer Build()
    {
        return new SequenceLayer(LayerTemplates.Select(definition => definition.Build()));
    }

    private static IEnumerable<INetworkLayerTemplate> Unwrap(
        IEnumerable<INetworkLayerTemplate> layerDefinitions)
    {
        if (layerDefinitions is null)
            return Enumerable.Empty<INetworkLayerTemplate>();

        return layerDefinitions.SelectMany(Unwrap);
    }

    private static IEnumerable<INetworkLayerTemplate> Unwrap(INetworkLayerTemplate layerDefinition)
    {
        if (layerDefinition is SequenceLayerTemplate sequenceDefinition)
            return sequenceDefinition.LayerTemplates;

        return layerDefinition.ToEnumerable();
    }
}
