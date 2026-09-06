using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.WeightInitialization;

using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Layers.NeuralLayers;

/// <summary>
/// A one-dimensional convolutional neural layer.
/// </summary>
/// <remarks>
/// A convolutional layer is a type of locally-connected layer in which connection weights and biases are shared.
/// <para>
/// This implementation of a convolutional layer is limited in functionality.  The number of input values must be
/// evenly divisible by the number of output values.  Each output value is connected to an equal number of
/// consecutive input values.  (The number of connected input values is referred to as the 'span'.)  The output 
/// values all share the same bias and connection weights.
/// </para><para>
/// For example, a convolutional layer with 32 input values and 8 output values would have a span of 4.  The first
/// output value would be connected to the first 4 input values; the second, to the next 4 input values, etc.
/// </para>
/// </remarks>
internal sealed class ConvolutionalLayer : NeuralLayer
{
    #region Construction

    public ConvolutionalLayer(
        int inputCount,
        int outputCount,
        IActivationFunction activationFunction,
        IConnectionWeightInitializer weightInitializer)
        : base(inputCount, outputCount, activationFunction, weightInitializer)
    {
        if (inputCount % outputCount != 0)
            throw new ArgumentException($"Parameter '{inputCount}' must be evenly divisible by '{outputCount}'.");

        WeightVector = new float[SpanCount];
    }

    [JsonConstructor]
    public ConvolutionalLayer(
        int inputCount,
        int outputCount,
        IActivationFunction activationFunction,
        IConnectionWeightInitializer weightInitializer,
        float[] weightVector,
        float bias)
        : base(inputCount, outputCount, activationFunction, weightInitializer)
    {
        WeightVector = weightVector;
        Bias = bias;
    }

    #endregion

    #region Public

    [JsonIgnore]
    public override string ShortDescription => $"(Convolutional, {OutputCount}, {ActivationFunction.ShortName})";

    /// <summary>
    /// Number of consecutive input values connected to each ouput value.
    /// </summary>
    [JsonIgnore]
    public int SpanCount => InputCount / OutputCount;

    #endregion

    #region Protected

    protected override void InitializeBiases(float bias = 0) => Bias = bias;

    protected override void InitializeWeights(IEnumerable<int> outgoingConnectionCounts)
    {
        // Weights are shared among all output values.  For the most conservative initial weight distribution,
        //  use the minimum fan-out value.
        int outgoingConnectionCount = outgoingConnectionCounts.Min();

        for (int index = 0; index < WeightVector.Length; index++)
            WeightVector[index] = WeightInitializer.GenerateWeight(SpanCount, outgoingConnectionCount);
    }

    protected override int GetInputConnectionCount(int inputIndex) => 1;

    protected override int GetOutputConnectionCount(int outputIndex) => SpanCount;

    protected override IEnumerable<float> GetInput(int outputIndex, IEnumerable<float> input)
    {
        return input.Skip(outputIndex * SpanCount).Take(SpanCount);
    }

    protected override float GetBias(int outputIndex) => Bias;

    protected override void UpdateBias(int outputIndex, float delta) => Bias += delta;

    protected override IEnumerable<float> GetWeights(int outputIndex) => WeightVector;

    protected override float GetWeight(int inputIndex, int outputIndex)
    {
        int startIndex = outputIndex * SpanCount;
        int endIndex = startIndex + SpanCount - 1;

        if (inputIndex < startIndex || inputIndex > endIndex)
            return 0;

        return WeightVector[inputIndex % SpanCount];
    }

    protected override void UpdateWeights(int outputIndex, IEnumerable<float> deltas)
    {
        int weightIndex = 0;

        foreach (float delta in deltas)
            WeightVector[weightIndex++] += delta;
    }

    #endregion

    #region Private

    [JsonInclude]
    private float[] WeightVector { get; }

    [JsonInclude]
    private float Bias { get; set; }

    #endregion
}