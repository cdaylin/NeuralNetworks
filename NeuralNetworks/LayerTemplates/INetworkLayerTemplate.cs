using Daylin.NeuralNetworks.Layers;

namespace Daylin.NeuralNetworks.LayerTemplates;

/// <summary>
/// A template (i.e.: descriptor, specification) for constructing a neural network layer.
/// </summary>
public interface INetworkLayerTemplate
{
    /// <summary>
    /// Number of values fed into the layer.
    /// </summary>
    int InputCount { get; }

    /// <summary>
    /// Number of values output from the layer.
    /// </summary>
    int OutputCount { get; }

    /// <summary>
    /// Constructs a layer as represented by this template.
    /// </summary>
    /// <returns>
    /// Returns a new network layer.
    /// </returns>
    INetworkLayer Build();
}
