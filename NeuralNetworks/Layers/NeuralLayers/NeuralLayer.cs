using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.WeightInitialization;
using Daylin.Utilities.Threading;

using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Layers.NeuralLayers;

/// <summary>
/// A layer of neurons in an artificial neural network.
/// </summary>
internal abstract class NeuralLayer : INetworkLayer
{
    #region Construction

    [JsonConstructor]
    public NeuralLayer(
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

    #endregion

    #region Public

    public override string ToString() => ShortDescription;

    public abstract string ShortDescription { get; }

    /// <summary>
    /// Number of values input to this layer.
    /// </summary>
    public int InputCount { get; }

    /// <summary>
    /// Number of values output from this layer (aka: number of neurons in this layer).
    /// </summary>
    public int OutputCount { get; }

    public virtual IEnumerable<int> Initialize(IEnumerable<int> outgoingConnectionCounts)
    {
        outgoingConnectionCounts.Count().ThrowIfNotEqualTo(OutputCount);

        InitializeBiases();
        InitializeWeights(outgoingConnectionCounts);

        return Enumerable.Range(0, InputCount).Select(GetInputConnectionCount);
    }

    public virtual async Task<IEnumerable<float>> CalculateOutputAsync(
        IEnumerable<float> input,
        CancellationToken cancellationToken = default)
    {
        float[] output = new float[OutputCount];

        await Parallel.ForAsync(0, OutputCount, cancellationToken,
            (outputIndex, cancellationToken) =>
            {
                output[outputIndex] = CalculateOutput(input, outputIndex);

                return ValueTask.CompletedTask;
            });

        return output;
    }

    public virtual async Task<(IEnumerable<float>, BackpropagateGradientsAsync)> FeedForwardAsync(
        IEnumerable<float> input,
        BackpropagateGradientsAsync? backpropagateGradientsAsync,
        CancellationToken cancellationToken = default)
    {
        float[] output = new float[OutputCount];
        float[] derivatives = new float[OutputCount];

        await Parallel.ForAsync(0, OutputCount, cancellationToken,
            (outputIndex, cancellationToken) =>
            {
                float weightedInput = CalculateWeightedInput(input, outputIndex);

                (float activation, float derivative) = ActivationFunction.Calculate(weightedInput);

                output[outputIndex] = activation;
                derivatives[outputIndex] = derivative;

                return ValueTask.CompletedTask;
            });

        return (output, (gradients, cancellationToken) =>
            BackpropagateGradientsAsync(
                gradients,
                input,
                output,
                derivatives,
                backpropagateGradientsAsync,
                cancellationToken));
    }

    #endregion

    #region Protected

    [JsonInclude]
    protected IActivationFunction ActivationFunction { get; }

    [JsonInclude]
    protected IConnectionWeightInitializer WeightInitializer { get; }

    /// <summary>
    /// The layer's parameters, grouped as the optimizer keeps state for them.
    /// </summary>
    /// <remarks>
    /// The groups hold the layer's own arrays by reference, so moving a group's parameters moves the layer's
    /// weights or biases.
    /// </remarks>
    internal IReadOnlyList<ParameterGroup> ParameterGroups => field ??= CreateParameterGroups();

    /// <summary>
    /// Creates the groups the layer's parameters are kept in.
    /// </summary>
    /// <remarks>
    /// Called once, when the groups are first needed.  An implementation returns groups holding its parameter
    /// arrays by reference, in an order it keeps for the life of the layer:  the optimizer's state is stored
    /// against that order and restored into it.
    /// </remarks>
    /// <returns>
    /// Returns the layer's parameter groups.
    /// </returns>
    protected abstract IReadOnlyList<ParameterGroup> CreateParameterGroups();

    /// <summary>
    /// Returns the gradients accumulated for a group of parameters.
    /// </summary>
    /// <param name="groupIndex">
    /// Index of the group, in the order <see cref="CreateParameterGroups"/> returned.
    /// </param>
    /// <returns>
    /// Returns the gradient accumulated for each parameter in the group.
    /// </returns>
    protected float[] GetGradients(int groupIndex) => ParameterGroups[groupIndex].Gradients;

    protected abstract void InitializeBiases(float bias = 0);

    protected abstract void InitializeWeights(IEnumerable<int> outgoingConnectionCounts);

    /// <summary>
    /// Returns the number of output values connected to an input value.
    /// </summary>
    /// <param name="inputIndex">
    /// Input value index.
    /// </param>
    /// <returns>
    /// Returns the number of output values connected to the input value.
    /// </returns>
    protected abstract int GetInputConnectionCount(int inputIndex);

    /// <summary>
    /// Returns the number of input values connected to an output value.
    /// </summary>
    /// <param name="outputIndex">
    /// Output value index.
    /// </param>
    /// <returns>
    /// Returns the number of input values connected to the output value.
    /// </returns>
    protected abstract int GetOutputConnectionCount(int outputIndex);

    protected abstract IEnumerable<float> GetInput(int outputIndex, IEnumerable<float> input);

    protected abstract float GetBias(int outputIndex);

    /// <summary>
    /// Accumulates the gradient of the cost with respect to the bias of an output value.
    /// </summary>
    /// <remarks>
    /// Where a layer shares one bias among several output values, every output value's gradient accumulates
    /// into that one bias.
    /// </remarks>
    /// <param name="outputIndex">
    /// Output value index.
    /// </param>
    /// <param name="gradient">
    /// The amount the bias is owed, before the learning rate.
    /// </param>
    protected abstract void AccumulateBiasGradient(int outputIndex, float gradient);

    /// <summary>
    /// Enumerates weights for connections between input values and an output value.
    /// </summary>
    /// <param name="outputIndex">
    /// Output value index.
    /// </param>
    /// <returns>
    /// Returns a connection weight for each input value connected to the output value.  The length of the
    /// sequence is equal to the result of <see cref="GetOutputConnectionCount(int)">
    /// GetOutputConnectionCount(outputIndex)</see>.
    /// </returns>
    protected abstract IEnumerable<float> GetWeights(int outputIndex);

    /// <summary>
    /// Returns the weight for a connection between an input value and an output value.
    /// </summary>
    /// <remarks>
    /// For an input value and output value that are not connected, the weight is always <c>0</c>.
    /// </remarks>
    /// <param name="inputIndex">
    /// Input value index.
    /// </param>
    /// <param name="outputIndex">
    /// Output value index.
    /// </param>
    /// <returns>
    /// Returns the weight for the connection between the input value and output value.  Returns <c>0</c> if the
    /// values are not connected.
    /// </returns>
    protected abstract float GetWeight(int inputIndex, int outputIndex);

    /// <summary>
    /// Accumulates the gradients of the cost with respect to the weights of the connections between input
    /// value.
    /// </summary>
    /// <remarks>
    /// Where a layer shares one set of weights among several output values, every output value's gradients
    /// accumulate into that one set.
    /// </remarks>
    /// <param name="outputIndex">
    /// Output value index.
    /// </param>
    /// <param name="gradients">
    /// The amount each connection weight is owed before the learning rate, with one value for each input value
    /// that is connected to the output value.
    /// </param>
    protected abstract void AccumulateWeightGradients(int outputIndex, IEnumerable<float> gradients);

    protected virtual float CalculateOutput(IEnumerable<float> input, int outputIndex)
    {
        float weightedInput = CalculateWeightedInput(input, outputIndex);

        return ActivationFunction.CalculateActivation(weightedInput);
    }

    protected virtual float CalculateWeightedInput(IEnumerable<float> input, int outputIndex)
    {
        float weightedSum = GetInput(outputIndex, input)
            .ElementwiseProduct(GetWeights(outputIndex))
            .Sum();

        return weightedSum + GetBias(outputIndex);
    }

    protected virtual async Task BackpropagateGradientsAsync(
        IEnumerable<float> gradients,
        IEnumerable<float> input,
        IReadOnlyList<float> output,
        IReadOnlyList<float> activationDerivatives,
        BackpropagateGradientsAsync? backpropagateGradientsAsync,
        CancellationToken cancellationToken)
    {
        // force enumeration, so that the gradients are calculated once here rather than once per reader
        IReadOnlyList<float> outputGradients = gradients.ElementwiseProduct(activationDerivatives).ToList();

        Task? backpropagationTask = null;

        if (backpropagateGradientsAsync is not null)
        {
            // force enumeration, so that the calculation completes on this thread rather than on the one
            //  backpropagating to the preceding layer
            IReadOnlyList<float> inputGradients = CalculateInputGradients(outputGradients).ToList();

            // force backpropagation to the preceding layer to run asynchronously, so that it will run
            //  concurrently with this layer's accumulation
            backpropagationTask = SynchronizationContexts.ThreadPoolContext.EnqueueAsync(
                () => backpropagateGradientsAsync(inputGradients, cancellationToken));
        }

        AccumulateGradients(input, output, outputGradients);

        if (backpropagationTask is not null)
            await backpropagationTask;
    }

    protected virtual IEnumerable<float> CalculateInputGradients(IReadOnlyList<float> outputGradients)
    {
        for (int inputIndex = 0; inputIndex < InputCount; inputIndex++)
        {
            float sum = 0;

            for (int outputIndex = 0; outputIndex < OutputCount; outputIndex++)
                sum += outputGradients[outputIndex] * GetWeight(inputIndex, outputIndex);

            yield return sum;
        }
    }

    /// <summary>
    /// Accumulates the gradient of the cost with respect to each of the layer's parameters, for one sample.
    /// </summary>
    /// <remarks>
    /// A gradient is accumulated rather than applied, so that a layer can learn from several samples before
    /// its parameters move.  A bias takes its output value's gradient; a connection weight takes that gradient
    /// multiplied by the input value the connection carries.
    /// </remarks>
    /// <param name="input">
    /// The input values the layer calculated its output from.
    /// </param>
    /// <param name="output">
    /// The output values the layer calculated.
    /// </param>
    /// <param name="outputGradients">
    /// The gradient of the cost with respect to each output value, multiplied by its activation derivative.
    /// </param>
    protected virtual void AccumulateGradients(
        IEnumerable<float> input,
        IReadOnlyList<float> output,
        IReadOnlyList<float> outputGradients)
    {
        outputGradients.Count.ThrowIfNotEqualTo(OutputCount);

        int outputIndex = 0;

        // for each output value
        foreach (float gradient in outputGradients)
        {
            AccumulateBiasGradient(outputIndex, gradient);

            // accumulate gradients for connection weights
            AccumulateWeightGradients(outputIndex, GetInput(outputIndex, input).ScalarProduct(gradient));

            outputIndex += 1;
        }
    }

    #endregion
}
