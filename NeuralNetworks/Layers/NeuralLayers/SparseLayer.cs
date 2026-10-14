using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.WeightInitialization;

using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Layers.NeuralLayers;

internal sealed class SparseLayer : NeuralLayer
{
    #region Private Static

    /// <summary>
    /// Refuses a mask that does not describe the layer it is masking.
    /// </summary>
    /// <remarks>
    /// Called from both constructors, since the invariant belongs to the layer rather than to one way of
    /// building it:  a document is as able to carry a mask of the wrong shape as a caller is to pass one.
    /// </remarks>
    /// <param name="connectionMask">
    /// The mask saying which connections the layer holds a weight for.
    /// </param>
    /// <param name="inputCount">
    /// Number of values fed into the layer.
    /// </param>
    /// <param name="outputCount">
    /// Number of values output from the layer.
    /// </param>
    private static void ThrowIfMaskDoesNotFitTheLayer(
        IConnectionMask connectionMask,
        int inputCount,
        int outputCount)
    {
        connectionMask.ThrowIfNull();
        connectionMask.InputCount.ThrowIfNotEqualTo(inputCount);
        connectionMask.OutputCount.ThrowIfNotEqualTo(outputCount);
    }

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
        ThrowIfMaskDoesNotFitTheLayer(connectionMask, inputCount, outputCount);

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
        // Checked here rather than where a weight is first read:  a document whose arrays do not match its mask
        //  otherwise builds a layer that answers and then fails partway through a training batch.  The mask's
        //  own rows are checked where the mask is read.
        ThrowIfMaskDoesNotFitTheLayer(connectionMask, inputCount, outputCount);
        biasVector.ThrowIfNull();
        biasVector.Length.ThrowIfNotEqualTo(outputCount);
        weightMatrix.ThrowIfNull();
        weightMatrix.ThrowIfContainsNull();
        weightMatrix.Length.ThrowIfNotEqualTo(outputCount);

        for (int outputIndex = 0; outputIndex < outputCount; outputIndex++)
        {
            weightMatrix[outputIndex].Length.ThrowIfNotEqualTo(
                connectionMask.GetUnmaskedConnectionCount(outputIndex));
        }

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

    protected override ParameterGroup CreateBiasGroup() => new(BiasVector);

    protected override IReadOnlyList<ParameterGroup> CreateWeightGroups() =>
        [.. WeightMatrix.Select(weights => new ParameterGroup(weights))];

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
