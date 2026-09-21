using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.WeightInitialization;

using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Layers.NeuralLayers;

/// <summary>
/// A fully-connected layer of neurons in an artificial neural network.
/// </summary>
/// <remarks>
/// A "dense" or "fully-connected" neural network layer has a connection from each input value to each output
/// value (i.e.: neuron).
/// </remarks>
internal sealed class DenseLayer : NeuralLayer
{
    #region Private Static

    /// <summary>
    /// Index of the parameter group holding the layer's biases.
    /// </summary>
    private static int BiasGroupIndex => 0;

    /// <summary>
    /// Index of the parameter group holding the connection weights of the first output value.
    /// </summary>
    /// <remarks>
    /// Each output value's weights are a group of their own, in output value order.
    /// </remarks>
    private static int FirstWeightGroupIndex => 1;

    #endregion

    #region Construction

    public DenseLayer(
        int inputCount,
        int outputCount,
        IActivationFunction activationFunction,
        IConnectionWeightInitializer weightInitializer)
        : base(inputCount, outputCount, activationFunction, weightInitializer)
    {
        BiasVector = new float[OutputCount];

        WeightMatrix = new float[OutputCount][];

        for (int index = 0; index < OutputCount; index++)
            WeightMatrix[index] = new float[InputCount];
    }

    [JsonConstructor]
    public DenseLayer(
        int inputCount,
        int outputCount,
        IActivationFunction activationFunction,
        IConnectionWeightInitializer weightInitializer,
        float[] biasVector,
        float[][] weightMatrix)
        : base(inputCount, outputCount, activationFunction, weightInitializer)
    {
        BiasVector = biasVector;
        WeightMatrix = weightMatrix;
    }

    #endregion

    #region Public

    [JsonIgnore]
    public override string ShortDescription => $"(Dense, {OutputCount}, {ActivationFunction.ShortName})";

    #endregion

    #region Protected

    protected override void InitializeBiases(float bias = 0) => BiasVector.Fill(bias);

    protected override void InitializeWeights(IEnumerable<int> outgoingConnectionCounts)
    {
        int outputIndex = 0;

        foreach (int outgoingConnectionCount in outgoingConnectionCounts)
            InitializeWeights(outputIndex++, outgoingConnectionCount);
    }

    protected override int GetInputConnectionCount(int inputIndex) => OutputCount;

    protected override int GetOutputConnectionCount(int outputIndex) => InputCount;

    protected override IEnumerable<float> GetInput(int outputIndex, IEnumerable<float> input) => input;

    protected override float GetBias(int outputIndex) => BiasVector[outputIndex];

    protected override IEnumerable<float> GetWeights(int outputIndex) => WeightMatrix[outputIndex];

    protected override float GetWeight(int inputIndex, int outputIndex) => WeightMatrix[outputIndex][inputIndex];

    protected override IReadOnlyList<ParameterGroup> CreateParameterGroups()
    {
        return [new ParameterGroup(BiasVector), .. WeightMatrix.Select(weights => new ParameterGroup(weights))];
    }

    protected override void AccumulateBiasGradient(int outputIndex, float gradient)
    {
        GetGradients(BiasGroupIndex)[outputIndex] += gradient;
    }

    protected override void AccumulateWeightGradients(int outputIndex, IEnumerable<float> gradients)
    {
        float[] weightGradients = GetGradients(FirstWeightGroupIndex + outputIndex);

        int weightIndex = 0;

        foreach (float gradient in gradients)
            weightGradients[weightIndex++] += gradient;
    }

    #endregion

    #region Private

    [JsonInclude]
    private float[] BiasVector { get; }

    [JsonInclude]
    private float[][] WeightMatrix { get; }

    private void InitializeWeights(int outputIndex, int outgoingConnectionCount)
    {
        int incomingConnectionCount = GetOutputConnectionCount(outputIndex);

        for (int inputIndex = 0; inputIndex < InputCount; inputIndex++)
        {
            WeightMatrix[outputIndex][inputIndex] =
                WeightInitializer.GenerateWeight(incomingConnectionCount, outgoingConnectionCount);
        }
    }

    #endregion
}
