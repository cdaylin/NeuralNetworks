using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.Layers;
using Daylin.NeuralNetworks.Layers.NeuralLayers;
using Daylin.NeuralNetworks.WeightInitialization;

namespace Daylin.NeuralNetworks.LayerTemplates;

public sealed class ConvolutionalLayerTemplate : NeuralLayerTemplate
{
    public ConvolutionalLayerTemplate(
        int inputCount,
        int outputCount,
        IActivationFunction activationFunction,
        IConnectionWeightInitializer weightInitializer)
        : base(inputCount, outputCount, activationFunction, weightInitializer)
    {
    }

    public override INetworkLayer Build()
    {
        return new ConvolutionalLayer(InputCount, OutputCount, ActivationFunction, WeightInitializer);
    }
}
