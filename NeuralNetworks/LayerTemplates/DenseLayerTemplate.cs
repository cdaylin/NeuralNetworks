using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.Layers;
using Daylin.NeuralNetworks.Layers.NeuralLayers;
using Daylin.NeuralNetworks.WeightInitialization;

namespace Daylin.NeuralNetworks.LayerTemplates;

public sealed class DenseLayerTemplate : NeuralLayerTemplate
{
    public DenseLayerTemplate(
        int inputCount,
        int outputCount,
        IActivationFunction activationFunction,
        IConnectionWeightInitializer weightInitializer)
        : base(inputCount, outputCount, activationFunction, weightInitializer)
    {
    }

    public override INetworkLayer Build()
    {
        return new DenseLayer(InputCount, OutputCount, ActivationFunction, WeightInitializer);
    }
}
