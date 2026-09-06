namespace Daylin.NeuralNetworks.Backpropagation;

/// <summary>
/// Data calculated during a forward pass through a neural network layer.
/// </summary>
/// <remarks>
/// In some cases it is more efficient to calculate both the activation and its derivative together.
/// When it is known that both types of data will be needed (e.g.: during a forward pass prior to
/// backpropagation of errors), request both values and store them until needed.
/// </remarks>
public readonly struct LayerActivationData
{
    public LayerActivationData()
    {
        Activations = Array.Empty<float>();
        ActivationDerivatives = Array.Empty<float>();
    }

    public LayerActivationData(IReadOnlyList<float> activations, IReadOnlyList<float> activationDerivatives)
    {
        activationDerivatives.ThrowIfNull();
        activations.Count.ThrowIfNotEqualTo(activationDerivatives.Count);

        Activations = activations;
        ActivationDerivatives = activationDerivatives;
    }

    /// <summary>
    /// Activation values for all neurons in the layer.
    /// </summary>
    /// <remarks>
    /// Each neuron's "activation" is calculated by applying an activation function to the "weighted input".
    /// </remarks>
    public IReadOnlyList<float> Activations { get; }

    /// <summary>
    /// Activation derivative values for all neurons in the layer.
    /// </summary>
    /// <remarks>
    /// Each neuron's "activation derivative" is the rate of change of the activation function with respect to
    /// the "weighted input".
    /// </remarks>
    public IReadOnlyList<float> ActivationDerivatives { get; }
}
