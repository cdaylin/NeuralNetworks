using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.WeightInitialization;

using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Layers.NeuralLayers;

internal sealed class SparseLayer : NeuralLayer
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
    /// Each output value's weights are a group of their own, in output value order, and a group holds only
    /// the weights of the connections the output value has.
    /// </remarks>
    private static int FirstWeightGroupIndex => 1;

    #endregion

    #region Construction

    public SparseLayer(
        int inputCount,
        int outputCount,
        IActivationFunction activationFunction,
        IConnectionWeightInitializer weightInitializer,
        IConnectionMask connectionMask)
        : base(inputCount, outputCount, activationFunction, weightInitializer)
    {
        connectionMask.ThrowIfNull();
        connectionMask.InputCount.ThrowIfNotEqualTo(inputCount);
        connectionMask.OutputCount.ThrowIfNotEqualTo(outputCount);

        ConnectionMask = connectionMask;

        BiasVector = new float[OutputCount];

        WeightMatrix = new float[OutputCount][];

        for (int outputIndex = 0; outputIndex < OutputCount; outputIndex++)
            WeightMatrix[outputIndex] = new float[GetOutputConnectionCount(outputIndex)];
    }

    [JsonConstructor]
    public SparseLayer(
        int inputCount,
        int outputCount,
        IActivationFunction activationFunction,
        IConnectionWeightInitializer weightInitializer,
        float[][] weightMatrix,
        float[] biasVector,
        IConnectionMask connectionMask)
        : base(inputCount, outputCount, activationFunction, weightInitializer)
    {
        WeightMatrix = weightMatrix;
        BiasVector = biasVector;
        ConnectionMask = connectionMask;
    }

    #endregion

    #region Public

    [JsonIgnore]
    public override string ShortDescription => $"(Sparse, {OutputCount}, {ActivationFunction.ShortName})";

    public IConnectionMask ConnectionMask { get; }

    #endregion

    #region Protected

    protected override void InitializeBiases(float bias = 0) => BiasVector.Fill(bias);

    protected override void InitializeWeights(IEnumerable<int> outgoingConnectionCounts)
    {
        int outputIndex = 0;

        foreach (int outgoingConnectionCount in outgoingConnectionCounts)
            InitializeWeights(outputIndex++, outgoingConnectionCount);
    }

    protected override int GetInputConnectionCount(int inputIndex)
    {
        return Enumerable.Range(0, OutputCount)
            .Count(outputIndex => !ConnectionMask.IsConnectionMasked(inputIndex, outputIndex));
    }

    protected override int GetOutputConnectionCount(int outputIndex)
    {
        return ConnectionMask.GetUnmaskedConnectionCount(outputIndex);
    }

    protected override IEnumerable<float> GetInput(int outputIndex, IEnumerable<float> input)
    {
        int inputIndex = 0;

        foreach (float inputValue in input)
        {
            if (!ConnectionMask.IsConnectionMasked(inputIndex, outputIndex))
                yield return inputValue;

            inputIndex++;
        }
    }

    protected override float GetBias(int outputIndex) => BiasVector[outputIndex];

    protected override IEnumerable<float> GetWeights(int outputIndex) => WeightMatrix[outputIndex];

    protected override IReadOnlyList<ParameterGroup> CreateParameterGroups()
    {
        return [new ParameterGroup(BiasVector), .. WeightMatrix.Select(weights => new ParameterGroup(weights))];
    }

    protected override void AccumulateBiasGradient(int outputIndex, float gradient)
    {
        GetGradients(BiasGroupIndex)[outputIndex] += gradient;
    }

    protected override float GetWeight(int inputIndex, int outputIndex)
    {
        if (ConnectionMask.IsConnectionMasked(inputIndex, outputIndex))
            return 0;

        int weightIndex = 0;

        for (int index = 0; index < inputIndex; index++)
        {
            if (!ConnectionMask.IsConnectionMasked(index, outputIndex))
                weightIndex += 1;
        }

        return WeightMatrix[outputIndex][weightIndex];
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

        float[] weights = WeightMatrix[outputIndex];

        for (int weightIndex = 0; weightIndex < weights.Length; weightIndex++)
        {
            WeightMatrix[outputIndex][weightIndex] =
                WeightInitializer.GenerateWeight(incomingConnectionCount, outgoingConnectionCount);
        }
    }

    #endregion
}
