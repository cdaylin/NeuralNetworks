using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.LayerTemplates;
using Daylin.NeuralNetworks.WeightInitialization;

namespace Daylin.NeuralNetworks;

/// <summary>
/// Responsible for building neural networks.
/// </summary>
/// <seealso cref="NeuralNetworkFactory"/>
public class NeuralNetworkBuilder
{
    #region Construction

    public NeuralNetworkBuilder(int inputCount)
    {
        inputCount.ThrowIfNotPositive();

        NetworkLayerTemplate = new InputLayerTemplate(inputCount);
    }

    public NeuralNetworkBuilder(INetworkLayerTemplate template)
    {
        template.ThrowIfNull();

        NetworkLayerTemplate = template;
    }

    #endregion

    #region Public

    public int InputCount => NetworkLayerTemplate.InputCount;

    public int OutputCount => NetworkLayerTemplate.OutputCount;

    public virtual INeuralNetwork BuildNetwork()
    {
        return NetworkFactory.CreateNetwork(NetworkLayerTemplate);
    }

    public virtual void AddDenseLayer(
        int outputCount,
        IActivationFunction activationFunction,
        IConnectionWeightInitializer weightInitializer)
    {
        outputCount.ThrowIfNotPositive();
        activationFunction.ThrowIfNull();
        weightInitializer.ThrowIfNull();

        AddLayer(NetworkFactory.CreateDenseLayer(
            NetworkLayerTemplate.OutputCount,
            outputCount,
            activationFunction,
            weightInitializer));
    }

    public virtual void AddSparseLayer(
        IConnectionMask connectionMask,
        IActivationFunction activationFunction,
        IConnectionWeightInitializer weightInitializer)
    {
        connectionMask.ThrowIfNull();
        connectionMask.InputCount.ThrowIfNotEqualTo(OutputCount);
        activationFunction.ThrowIfNull();
        weightInitializer.ThrowIfNull();

        AddLayer(NetworkFactory.CreateSparseLayer(
            connectionMask,
            activationFunction,
            weightInitializer));
    }

    public virtual void AddConvolutionalLayer(
        int outputCount,
        IActivationFunction activationFunction,
        IConnectionWeightInitializer weightInitializer)
    {
        outputCount.ThrowIfNotPositive();
        outputCount.ThrowIfGreaterThanOrEqualTo(OutputCount);
        activationFunction.ThrowIfNull();
        weightInitializer.ThrowIfNull();

        if (OutputCount % outputCount != 0)
        {
            throw new ArgumentException(
                $"'{OutputCount}' must be evenly divisible by parameter '{outputCount}'.",
                nameof(outputCount));
        }

        AddLayer(NetworkFactory.CreateConvolutionalLayer(
            OutputCount,
            outputCount,
            activationFunction,
            weightInitializer));
    }

    public virtual void AddConnectionRoutingLayer(IndexMap connectionMap)
    {
        connectionMap.ThrowIfNull();
        connectionMap.Length.ThrowIfNotEqualTo(OutputCount);

        AddLayer(NetworkFactory.CreateConnectionRoutingLayer(connectionMap));
    }

    #endregion

    #region Protected

    protected NeuralNetworkFactory NetworkFactory { get; } = new();

    protected INetworkLayerTemplate NetworkLayerTemplate { get; set; }

    protected virtual void AddLayer(INetworkLayerTemplate layerTemplate)
    {
        NetworkLayerTemplate = NetworkFactory.CreateSequence(NetworkLayerTemplate, layerTemplate);
    }

    protected virtual void AppendLayer(INetworkLayerTemplate layerTemplate)
    {
        NetworkLayerTemplate = NetworkFactory.Concatenate(NetworkLayerTemplate, layerTemplate);
    }

    #endregion
}
