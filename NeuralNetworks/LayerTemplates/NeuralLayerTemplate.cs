using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.Layers;
using Daylin.NeuralNetworks.WeightInitialization;

namespace Daylin.NeuralNetworks.LayerTemplates;

public abstract class NeuralLayerTemplate : INetworkLayerTemplate
{
    protected NeuralLayerTemplate(
        int inputCount,
        int outputCount,
        IActivationFunction activationFunction,
        IConnectionWeightInitializer weightInitializer)
    {
        inputCount.ThrowIfNotPositive();
        outputCount.ThrowIfNotPositive();
        activationFunction.ThrowIfNull();
        weightInitializer.ThrowIfNull();

        InputCount = inputCount;
        OutputCount = outputCount;
        ActivationFunction = activationFunction;
        WeightInitializer = weightInitializer;
    }

    public int InputCount { get; }

    public int OutputCount { get; }

    public IActivationFunction ActivationFunction { get; }

    public IConnectionWeightInitializer WeightInitializer { get; }

    public abstract INetworkLayer Build();
}
