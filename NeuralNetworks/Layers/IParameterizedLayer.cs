using Daylin.NeuralNetworks.Layers.NeuralLayers;

namespace Daylin.NeuralNetworks.Layers;

/// <summary>
/// A layer that holds parameters a training session can move, directly or through the layers it contains.
/// </summary>
/// <remarks>
/// Implementing this is what puts a layer's parameters in reach of training:  a layer that holds parameters and
/// does not implement it feeds forward and backpropagates correctly and never learns.  Nothing can enforce that,
/// since a layer type declares what it implements;  what this interface buys is that the obligation sits on the
/// layer, beside the parameters, rather than in a list of layer types kept by the network.
/// </remarks>
internal interface IParameterizedLayer
{
    /// <summary>
    /// The groups the layer's parameters are kept in, in an order it keeps for the life of the layer.
    /// </summary>
    /// <remarks>
    /// The order is what optimizer state is exported and imported under, so it cannot change while a session
    /// that saved state for this layer might still be resumed.
    /// </remarks>
    IReadOnlyList<ParameterGroup> ParameterGroups { get; }
}
