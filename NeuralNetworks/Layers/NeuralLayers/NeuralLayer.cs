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

    public virtual async Task<(IEnumerable<float>, BackpropagateErrorsAsync)> FeedForwardAsync(
        IEnumerable<float> input,
        BackpropagateErrorsAsync? backpropagateErrorsAsync,
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

        return (output, (learningRate, errors, cancellationToken) =>
            BackpropagateErrorsAsync(
                learningRate,
                errors,
                input,
                output,
                derivatives,
                backpropagateErrorsAsync,
                cancellationToken));
    }

    #endregion

    #region Protected

    [JsonInclude]
    protected IActivationFunction ActivationFunction { get; }

    [JsonInclude]
    protected IConnectionWeightInitializer WeightInitializer { get; }

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

    protected abstract void UpdateBias(int outputIndex, float delta);

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
    /// Updates weights for connections between input values and an output value.
    /// </summary>
    /// <param name="outputIndex">
    /// Output value index.
    /// </param>
    /// <param name="deltas">
    /// Sequence of values to add to connection weights, with one value for each input value that is connected
    /// to the output value.
    /// </param>
    protected abstract void UpdateWeights(int outputIndex, IEnumerable<float> deltas);

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

    protected virtual async Task BackpropagateErrorsAsync(
        float learningRate,
        IEnumerable<float> errors,
        IEnumerable<float> input,
        IReadOnlyList<float> output,
        IReadOnlyList<float> activationDerivatives,
        BackpropagateErrorsAsync? backpropagateErrorsAsync,
        CancellationToken cancellationToken)
    {
        // force enumeration, so that error calculations complete before weights are updated
        IReadOnlyList<float> outputErrors = errors.ElementwiseProduct(activationDerivatives).ToList();

        Task? backpropagationTask = null;

        if (backpropagateErrorsAsync is not null)
        {
            // force enumeration, so that error calculations complete before weights are updated
            IReadOnlyList<float> inputErrors = CalculateInputErrors(outputErrors).ToList();

            // force backpropagation to the preceding layer to run asynchronously, so that it will run
            //  concurrently with weight updates
            backpropagationTask = SynchronizationContexts.ThreadPoolContext.EnqueueAsync(
                () => backpropagateErrorsAsync(learningRate, inputErrors, cancellationToken));
        }

        UpdateWeightsAndBiases(
            learningRate,
            input,
            output,
            outputErrors);

        if (backpropagationTask is not null)
            await backpropagationTask;
    }

    protected virtual IEnumerable<float> CalculateInputErrors(IReadOnlyList<float> outputErrors)
    {
        for (int inputIndex = 0; inputIndex < InputCount; inputIndex++)
        {
            float sum = 0;

            for (int outputIndex = 0; outputIndex < OutputCount; outputIndex++)
                sum += outputErrors[outputIndex] * GetWeight(inputIndex, outputIndex);

            yield return sum;
        }
    }

    protected virtual void UpdateWeightsAndBiases(
        float learningRate,
        IEnumerable<float> input,
        IReadOnlyList<float> output,
        IReadOnlyList<float> outputErrors)
    {
        outputErrors.Count.ThrowIfNotEqualTo(OutputCount);

        int outputIndex = 0;

        // for each output value
        foreach (float error in outputErrors)
        {
            float deltaFactor = learningRate * error;

            UpdateBias(outputIndex, deltaFactor);

            // update connection weights
            UpdateWeights(outputIndex, GetInput(outputIndex, input).ScalarProduct(deltaFactor));

            outputIndex += 1;
        }
    }

    #endregion
}
