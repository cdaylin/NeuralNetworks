using Daylin.NeuralNetworks.Layers.CompositeLayers;
using Daylin.NeuralNetworks.Layers.NeuralLayers;
using Daylin.NeuralNetworks.Layers.TransformationLayers;

using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Layers;

/// <summary>
/// A delegate for backpropagation of errors.
/// </summary>
/// <remarks>
/// The delegate has two responsibilities.  Firstly, it should update the layer (e.g.: adjust connection weights
/// and biases) with the aim of reducing errors for future input samples.  Secondly, it must continue to
/// backpropagate errors by invoking the delegate obtained from the previous layer in the network, if one exists.
/// For improved performance, these two responsibilities should be executed concurrently when practicable.
/// </remarks>
/// <param name="learningRate">
/// The learning rate (a factor used to control the rate at which connection weights and biases are adjusted in
/// response to errors).
/// </param>
/// <param name="errors">
/// The error attributed to each output value.  The count must equal the layer's 
/// <see cref="INetworkLayer.OutputCount"/>.
/// </param>
/// <param name="cancellationToken">
/// A cancellation token.
/// </param>
/// <returns>
/// Returns a task representing this <c>async</c> operation.
/// </returns>
public delegate Task BackpropagateErrorsAsync(
    float learningRate,
    IEnumerable<float> errors,
    CancellationToken cancellationToken);

[JsonPolymorphic]
[JsonDerivedType(typeof(DenseLayer), nameof(DenseLayer))]
[JsonDerivedType(typeof(SparseLayer), nameof(SparseLayer))]
[JsonDerivedType(typeof(ConvolutionalLayer), nameof(ConvolutionalLayer))]
[JsonDerivedType(typeof(SequenceLayer), nameof(SequenceLayer))]
[JsonDerivedType(typeof(ConcatenationLayer), nameof(ConcatenationLayer))]
[JsonDerivedType(typeof(IdentityTransformationLayer), nameof(IdentityTransformationLayer))]
[JsonDerivedType(typeof(ConnectionRoutingLayer), nameof(ConnectionRoutingLayer))]
public interface INetworkLayer
{
    /// <summary>
    /// A succinct human-readable description of the layer.
    /// </summary>
    string ShortDescription { get; }

    /// <summary>
    /// Number of values fed into the layer.
    /// </summary>
    int InputCount { get; }

    /// <summary>
    /// Number of values output from the layer.
    /// </summary>
    int OutputCount { get; }

    /// <summary>
    /// Initializes the layer (e.g.: connection weights and biases).
    /// </summary>
    /// <remarks>
    /// Calls to this method are intended to be initiated only by the neural network containing the layer.
    /// The method is invoked one time only, when a network containing the layer is first created.
    /// <para>
    /// Parameter <paramref name="outgoingConnectionCounts"/> provides the number of connections from each output
    /// value in this layer to output values in the subsequent layer in the network.  This value is often
    /// referred to as "fan-out" in artificial neural network literature.
    /// </para><para>
    /// The return value provides the number of connections from each input value to each output value in this
    /// layer.  This value is often referred to as "fan-in" in artificial neural network literature.  A 
    /// "transformation layer" (a layer that transforms values but does not contain neurons) should return
    /// <paramref name="outgoingConnectionCounts"/>, which is the "fan-in" for the next neural layer in the
    /// network.
    /// </para>
    /// </remarks>
    /// <param name="outgoingConnectionCounts">
    /// The number of connections from each output value in this layer to output values in the subsequent 
    /// layer in the network.  The sequence has a count equal to <see cref="OutputCount"/>.  Each value in
    /// the sequence is in the range [0, <see cref="InputCount"/>].
    /// </param>
    /// <returns>
    /// Returns the number of connections from each input value to each output value in this layer.  The
    /// returned sequence has a count equal to <see cref="InputCount"/>, and each value in the sequence is in
    /// the range [0, <see cref="OutputCount"/>].
    /// </returns>
    IEnumerable<int> Initialize(IEnumerable<int> outgoingConnectionCounts);

    /// <summary>
    /// Calculates output values for a set of input values.
    /// </summary>
    /// <param name="input">
    /// Input values.  The length must equal <see cref="InputCount"/>.
    /// </param>
    /// <param name="cancellationToken">
    /// A cancellation token.
    /// </param>
    /// <returns>
    /// Returns a list of output values.  The length must equal <see cref="OutputCount"/>.
    /// </returns>
    Task<IEnumerable<float>> CalculateOutputAsync(
        IEnumerable<float> input,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Calculates output values for a set of input values, and provides a delegate for the backpropagation of
    /// errors.
    /// </summary>
    /// <param name="input">
    /// Input values.  The length must equal <see cref="InputCount"/>.
    /// </param>
    /// <param name="backpropagateErrorsAsync">
    /// A delegate for backpropagating errors to the preceeding layer in the network.
    /// </param>
    /// <param name="cancellationToken">
    /// A cancellation token.
    /// </param>
    /// <returns>
    /// Returns two values.  First is a list of output values, with length equal to <see cref="OutputCount"/>.
    /// Second is a delegate for backpropagating errors to this layer.
    /// </returns>
    Task<(IEnumerable<float>, BackpropagateErrorsAsync)> FeedForwardAsync(
        IEnumerable<float> input,
        BackpropagateErrorsAsync? backpropagateErrorsAsync,
        CancellationToken cancellationToken = default);
}
