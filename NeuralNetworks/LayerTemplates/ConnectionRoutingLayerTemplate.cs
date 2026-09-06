using Daylin.NeuralNetworks.Layers;
using Daylin.NeuralNetworks.Layers.TransformationLayers;

namespace Daylin.NeuralNetworks.LayerTemplates;

public class ConnectionRoutingLayerTemplate : INetworkLayerTemplate
{
    public ConnectionRoutingLayerTemplate(IndexMap connectionMap)
    {
        connectionMap.ThrowIfNull();

        ConnectionMap = connectionMap;
    }

    public int InputCount => ConnectionMap.Length;

    public int OutputCount => ConnectionMap.Length;

    public INetworkLayer Build() => new ConnectionRoutingLayer(ConnectionMap);

    private IndexMap ConnectionMap { get; }
}
