using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.WeightInitialization;
using Daylin.Utilities.Threading;

using System.Runtime.ExceptionServices;
using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Layers.NeuralLayers;

/// <summary>
/// A layer of neurons in an artificial neural network.
/// </summary>
internal abstract class NeuralLayer : INetworkLayer, IParameterizedLayer
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

    #region Explicit Interface Implementations

    /// <inheritdoc cref="IParameterizedLayer.ParameterGroups"/>
    /// <remarks>
    /// The biases first, then the connection weights:  that order is the layer's contract with a training
    /// session, which exports and imports optimizer state under it, so it is composed here rather than left to
    /// each layer.  The groups hold the layer's own arrays by reference, so moving a group's parameters moves
    /// the layer's weights or biases.
    /// <para>
    /// Implemented explicitly, so that the serializer does not see it:  an implicit implementation would be
    /// public, and a saved network carries no training state.
    /// </para>
    /// </remarks>
    IReadOnlyList<ParameterGroup> IParameterizedLayer.ParameterGroups =>
        field ??= [BiasGroup, .. WeightGroups];

    #endregion

    #region Protected

    [JsonInclude]
    protected IActivationFunction ActivationFunction { get; }

    [JsonInclude]
    protected IConnectionWeightInitializer WeightInitializer { get; }

    /// <summary>
    /// The group holding the layer's biases.
    /// </summary>
    /// <remarks>
    /// Created once, when the groups are first needed.
    /// </remarks>
    protected ParameterGroup BiasGroup => field ??= CreateBiasGroup();

    /// <summary>
    /// The groups holding the layer's connection weights, one for each output value whose weights are its own,
    /// or a single group where every output value shares one set.
    /// </summary>
    /// <remarks>
    /// Created once, when the groups are first needed.
    /// </remarks>
    protected IReadOnlyList<ParameterGroup> WeightGroups => field ??= CreateWeightGroups();

    /// <summary>
    /// Creates the group holding the layer's biases.
    /// </summary>
    /// <remarks>
    /// Called once, when the groups are first needed.  An implementation returns a group holding its parameter
    /// array by reference, so that moving the group's parameters moves the layer's own.
    /// </remarks>
    /// <returns>
    /// Returns the group holding the layer's biases.
    /// </returns>
    protected abstract ParameterGroup CreateBiasGroup();

    /// <summary>
    /// Creates the groups the layer's connection weights are kept in.
    /// </summary>
    /// <remarks>
    /// Called once, when the groups are first needed.  An implementation returns one group for each output
    /// value whose weights are its own, or a single group where every output value shares one set of weights,
    /// and overrides <see cref="GetWeightGroup"/> to say which group an output value's gradients belong to.
    /// </remarks>
    /// <returns>
    /// Returns the groups holding the layer's connection weights, in output value order.
    /// </returns>
    protected abstract IReadOnlyList<ParameterGroup> CreateWeightGroups();

    /// <summary>
    /// Returns the group holding the connection weights an output value's gradients belong to.
    /// </summary>
    /// <param name="outputIndex">
    /// Index of the output value.
    /// </param>
    /// <returns>
    /// Returns that output value's group of connection weights.
    /// </returns>
    protected virtual ParameterGroup GetWeightGroup(int outputIndex) => WeightGroups[outputIndex];

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
    /// The gradient of the cost with respect to that bias, for one sample.
    /// </param>
    protected virtual void AccumulateBiasGradient(int outputIndex, float gradient) =>
        BiasGroup.AccumulateGradient(outputIndex, gradient);

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
    /// Accumulates the gradients of the cost with respect to the weights of the connections between the input
    /// values and an output value.
    /// </summary>
    /// <remarks>
    /// Where a layer shares one set of weights among several output values, every output value's gradients
    /// accumulate into that one set.
    /// </remarks>
    /// <param name="outputIndex">
    /// Output value index.
    /// </param>
    /// <param name="gradients">
    /// The gradient of the cost with respect to each of those connection weights, for one sample, with one
    /// value for each input value that is connected to the output value.
    /// </param>
    protected virtual void AccumulateWeightGradients(int outputIndex, IEnumerable<float> gradients) =>
        GetWeightGroup(outputIndex).AccumulateGradients(gradients);

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

        // The preceding layers must have finished writing their gradients before this returns, however this
        //  layer's accumulation ends:  a batch abandoned partway through discards the gradients accumulated so
        //  far, and a task still running would write into the layers it had just cleared.  The two run
        //  concurrently, so where both fail nothing here establishes an order between them;  this layer's own
        //  failure is reported because it is the one describing the work this call was asked to do.
        ExceptionDispatchInfo? accumulationFailure = null;

        try
        {
            AccumulateGradients(input, outputGradients);
        }
        catch (Exception exception)
        {
            accumulationFailure = ExceptionDispatchInfo.Capture(exception);
        }

        if (backpropagationTask is not null)
        {
            try
            {
                await backpropagationTask;
            }
            catch when (accumulationFailure is not null)
            {
                // Dropped in favour of this layer's own failure, thrown below.
            }
        }

        accumulationFailure?.Throw();
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
    /// <param name="outputGradients">
    /// The gradient of the cost with respect to each output value, multiplied by its activation derivative.
    /// </param>
    protected virtual void AccumulateGradients(
        IEnumerable<float> input,
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
