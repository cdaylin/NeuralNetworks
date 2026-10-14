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
    #region Private Static

    /// <summary>
    /// Refuses a layer whose input values do not divide evenly among its output values.
    /// </summary>
    /// <remarks>
    /// Every output value spans the same number of input values, so a remainder would leave input values no
    /// output value is connected to.  Called from both constructors, since the invariant belongs to the layer
    /// rather than to one way of building it.
    /// </remarks>
    /// <param name="inputCount">
    /// Number of values fed into the layer.
    /// </param>
    /// <param name="outputCount">
    /// Number of values output from the layer.
    /// </param>
    /// <exception cref="ArgumentException">
    /// The input count is not a whole multiple of the output count.
    /// </exception>
    private static void ThrowIfSpansDoNotDivide(int inputCount, int outputCount)
    {
        if (inputCount % outputCount == 0)
            return;

        throw new ArgumentException(
            $"A convolutional layer spans its {inputCount} input values evenly across its {outputCount} output "
            + $"values, which leaves {inputCount % outputCount} of them unconnected.",
            nameof(inputCount));
    }

    #endregion

    #region Construction

    public ConvolutionalLayer(
        int inputCount,
        int outputCount,
        IActivationFunction activationFunction,
        IConnectionWeightInitializer weightInitializer)
        : base(inputCount, outputCount, activationFunction, weightInitializer)
    {
        ThrowIfSpansDoNotDivide(inputCount, outputCount);

        WeightVector = new float[SpanCount];
        BiasVector = new float[1];
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
        // The same two checks the other constructor makes, because a document is as able to disagree with the
        //  counts beside it as a caller is:  inputs that do not divide among the outputs would build a layer
        //  that silently never connects the remainder, and a weight vector that does not span the layer would
        //  build one that answers and then fails partway through a training batch.
        ThrowIfSpansDoNotDivide(inputCount, outputCount);
        weightVector.ThrowIfNull();
        weightVector.Length.ThrowIfNotEqualTo(SpanCount);

        WeightVector = weightVector;
        BiasVector = [bias];
    }

    #endregion

    #region Public

    [JsonIgnore]
    public override string ShortDescription => $"(Convolutional, {OutputCount}, {ActivationFunction.ShortName})";

    /// <summary>
    /// Number of consecutive input values connected to each output value.
    /// </summary>
    [JsonIgnore]
    public int SpanCount => InputCount / OutputCount;

    #endregion

    #region Protected

    protected override void InitializeBiases(float bias = 0) => BiasVector[0] = bias;

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

    protected override IEnumerable<float> GetWeights(int outputIndex) => WeightVector;

    protected override float GetWeight(int inputIndex, int outputIndex)
    {
        int startIndex = outputIndex * SpanCount;
        int endIndex = startIndex + SpanCount - 1;

        if (inputIndex < startIndex || inputIndex > endIndex)
            return 0;

        return WeightVector[inputIndex % SpanCount];
    }

    protected override ParameterGroup CreateBiasGroup() => new(BiasVector);

    protected override IReadOnlyList<ParameterGroup> CreateWeightGroups() => [new ParameterGroup(WeightVector)];

    // Every output value shares the one set of weights, so every span's gradients accumulate into it.
    protected override ParameterGroup GetWeightGroup(int outputIndex) => WeightGroups[0];

    protected override void AccumulateBiasGradient(int outputIndex, float gradient)
    {
        // Every output value shares the one bias, so every output value's gradient accumulates into it.
        BiasGroup.AccumulateGradient(0, gradient);
    }

    #endregion

    #region Private

    [JsonInclude]
    private float[] WeightVector { get; }

    /// <summary>
    /// The bias shared by every output value, held as a vector of one value so that it forms a parameter
    /// group like any other.
    /// </summary>
    [JsonIgnore]
    private float[] BiasVector { get; }

    /// <summary>
    /// The bias shared by every output value.
    /// </summary>
    [JsonInclude]
    private float Bias => BiasVector[0];

    #endregion
}