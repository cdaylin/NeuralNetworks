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
        // Checked here rather than where a weight is first read:  a document whose arrays do not fit its counts
        //  otherwise builds a layer that answers and then fails partway through a training batch.
        biasVector.ThrowIfNull();
        biasVector.Length.ThrowIfNotEqualTo(outputCount);
        weightMatrix.ThrowIfNull();
        weightMatrix.ThrowIfContainsNull();
        weightMatrix.Length.ThrowIfNotEqualTo(outputCount);

        foreach (float[] weights in weightMatrix)
            weights.Length.ThrowIfNotEqualTo(inputCount);

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

    protected override ParameterGroup CreateBiasGroup() => new(BiasVector);

    protected override IReadOnlyList<ParameterGroup> CreateWeightGroups() =>
        [.. WeightMatrix.Select(weights => new ParameterGroup(weights))];

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
