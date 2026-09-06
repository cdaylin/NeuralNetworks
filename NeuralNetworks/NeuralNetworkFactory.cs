using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.LayerTemplates;
using Daylin.NeuralNetworks.WeightInitialization;

namespace Daylin.NeuralNetworks;

/// <summary>
/// Responsible for constructing neural network layers and assembling them into a neural network.
/// </summary>
/// <seealso cref="NeuralNetworkBuilder"/>
public class NeuralNetworkFactory
{
    #region Construction

    public NeuralNetworkFactory()
    {
    }

    #endregion

    #region Public

    public virtual INeuralNetwork CreateNetwork(params IEnumerable<INetworkLayerTemplate> layers)
    {
        layers.ThrowIfNull();
        layers.ThrowIfContainsNull();
        layers.ThrowIfEmpty();

        INetworkLayerTemplate layerTemplate = layers.Count() > 1 ? CreateSequence(layers) : layers.First();

        NeuralNetwork network = new(layerTemplate.Build());

        network.Initialize();

        return network;
    }

    public virtual INetworkLayerTemplate CreateSequence(params IEnumerable<INetworkLayerTemplate> layers)
    {
        layers.ThrowIfNull();
        layers.ThrowIfContainsNull();
        layers.ThrowIfEmpty();
        layers.Count().ThrowIfLessThan(2);

        return new SequenceLayerTemplate(layers);
    }

    public virtual INetworkLayerTemplate Concatenate(params IEnumerable<INetworkLayerTemplate> layers)
    {
        layers.ThrowIfNull();
        layers.ThrowIfContainsNull();
        layers.ThrowIfEmpty();
        layers.Count().ThrowIfLessThan(2);

        return new ConcatenationLayerTemplate(layers);
    }

    public virtual INetworkLayerTemplate ConcatenateInput(INetworkLayerTemplate layer, int inputCount)
    {
        layer.ThrowIfNull();
        inputCount.ThrowIfNotPositive();

        return Concatenate(layer, CreateInputLayer(inputCount));
    }

    public virtual INetworkLayerTemplate CreateDenseLayer(
        int inputCount,
        int outputCount,
        IActivationFunction activationFunction,
        IConnectionWeightInitializer weightInitializer)
    {
        inputCount.ThrowIfNotPositive();
        outputCount.ThrowIfNotPositive();
        activationFunction.ThrowIfNull();
        weightInitializer.ThrowIfNull();

        return new DenseLayerTemplate(
            inputCount,
            outputCount,
            activationFunction,
            weightInitializer);
    }

    public virtual INetworkLayerTemplate CreateSparseLayer(
        IConnectionMask connectionMask,
        IActivationFunction activationFunction,
        IConnectionWeightInitializer weightInitializer)
    {
        connectionMask.ThrowIfNull();
        activationFunction.ThrowIfNull();
        weightInitializer.ThrowIfNull();

        return new SparseLayerTemplate(connectionMask, activationFunction, weightInitializer);
    }

    public virtual INetworkLayerTemplate CreateConvolutionalLayer(
        int inputCount,
        int outputCount,
        IActivationFunction activationFunction,
        IConnectionWeightInitializer weightInitializer)
    {
        inputCount.ThrowIfNotPositive();
        outputCount.ThrowIfNotPositive();
        outputCount.ThrowIfGreaterThanOrEqualTo(inputCount);
        activationFunction.ThrowIfNull();
        weightInitializer.ThrowIfNull();

        if (inputCount % outputCount != 0)
        {
            throw new ArgumentException(
                $"Parameter '{inputCount}' must be evenly divisible by parameter '{outputCount}'.");
        }

        return new ConvolutionalLayerTemplate(
            inputCount,
            outputCount,
            activationFunction,
            weightInitializer);
    }

    public virtual INetworkLayerTemplate CreateConnectionRoutingLayer(IndexMap connectionMap)
    {
        connectionMap.ThrowIfNull();

        return new ConnectionRoutingLayerTemplate(connectionMap);
    }

    public virtual INetworkLayerTemplate CreatePassthroughLayer(int inputCount)
    {
        inputCount.ThrowIfNotPositive();

        return new PassthroughLayerTemplate(inputCount);
    }

    #endregion

    #region Protected

    protected virtual INetworkLayerTemplate CreateInputLayer(int inputCount)
    {
        inputCount.ThrowIfNotPositive();

        return new InputLayerTemplate(inputCount);
    }

    #endregion
}
