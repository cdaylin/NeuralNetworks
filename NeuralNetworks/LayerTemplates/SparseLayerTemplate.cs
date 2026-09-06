using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.Layers;
using Daylin.NeuralNetworks.Layers.NeuralLayers;
using Daylin.NeuralNetworks.WeightInitialization;

using System.Diagnostics;

namespace Daylin.NeuralNetworks.LayerTemplates;

internal class SparseLayerTemplate : NeuralLayerTemplate
{
    public SparseLayerTemplate(
        IConnectionMask connectionMask,
        IActivationFunction activationFunction,
        IConnectionWeightInitializer weightInitializer)
        : base(ValidateConnectionMask(connectionMask).InputCount,
              connectionMask.OutputCount,
              activationFunction,
              weightInitializer)
    {
        ConnectionMask = connectionMask;
    }

    public IConnectionMask ConnectionMask { get; }

    /// <summary>
    /// Constructs a network component according to this definition.
    /// </summary>
    /// <returns>
    /// Returns a new network component.
    /// </returns>
    public override INetworkLayer Build()
    {
        return new SparseLayer(InputCount, OutputCount, ActivationFunction, WeightInitializer, ConnectionMask);
    }

    [StackTraceHidden]
    private static IConnectionMask ValidateConnectionMask(IConnectionMask connectionMask)
    {
        connectionMask.ThrowIfNull();

        return connectionMask;
    }
}
