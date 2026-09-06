using Daylin.NeuralNetworks.Layers;
using Daylin.NeuralNetworks.Layers.TransformationLayers;

namespace Daylin.NeuralNetworks.LayerTemplates;

public sealed class PassthroughLayerTemplate : INetworkLayerTemplate
{
    public PassthroughLayerTemplate(int inputCount)
    {
        inputCount.ThrowIfNotPositive();

        InputCount = inputCount;
    }

    public int InputCount { get; }

    public int OutputCount => InputCount;

    public INetworkLayer Build()
    {
        return new IdentityTransformationLayer(InputCount, "Passthrough");
    }
}
