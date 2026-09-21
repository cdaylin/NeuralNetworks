namespace Daylin.NeuralNetworks.Backpropagation;

/// <summary>
/// One hyperbolic tangent neuron, in double precision, against which the library's training is judged.
/// </summary>
/// <remarks>
/// The neuron calculates what a dense layer of one output value calculates, and the gradient of the cost with
/// respect to each of its parameters for a sample:  the bias takes the output value's cost derivative
/// multiplied by the activation derivative, and each connection weight that same quantity multiplied by the
/// input value it carries.  A parameter moves <em>against</em> its gradient.
/// <para>
/// A test using this holds the optimizer's arithmetic itself, so that what is compared is the library's result
/// against the formulas the optimizers document, rather than against the library's own way of reaching them.
/// </para>
/// </remarks>
internal sealed class ReferenceNeuron
{
    #region Construction

    public ReferenceNeuron(IEnumerable<double> weights, double bias)
    {
        weights.ThrowIfNull();

        Weights = weights.ToArray();
        Bias = bias;
    }

    #endregion

    #region Public

    /// <summary>
    /// The neuron's connection weights.
    /// </summary>
    public double[] Weights { get; }

    /// <summary>
    /// The neuron's bias.
    /// </summary>
    public double Bias { get; set; }

    /// <summary>
    /// Calculates the neuron's output for an input.
    /// </summary>
    /// <param name="input">
    /// Input values, one for each connection weight.
    /// </param>
    /// <returns>
    /// Returns the neuron's output value.
    /// </returns>
    public double CalculateOutput(IReadOnlyList<double> input)
    {
        input.Count.ThrowIfNotEqualTo(Weights.Length);

        double weightedInput = Bias;

        for (int index = 0; index < Weights.Length; index++)
            weightedInput += Weights[index] * input[index];

        return Math.Tanh(weightedInput);
    }

    /// <summary>
    /// Calculates the gradient of the cost with respect to each of the neuron's parameters, for one sample.
    /// </summary>
    /// <param name="input">
    /// Input values.
    /// </param>
    /// <param name="target">
    /// The desired output value.
    /// </param>
    /// <returns>
    /// Returns the gradients for the connection weights, followed by the one for the bias.
    /// </returns>
    public (IReadOnlyList<double> WeightGradients, double BiasGradient) CalculateGradients(
        IReadOnlyList<double> input,
        double target)
    {
        double activation = CalculateOutput(input);

        double gradient = (activation - target) * (1 - (activation * activation));

        return (input.Select(value => gradient * value).ToList(), gradient);
    }

    /// <summary>
    /// Calculates the gradients a batch of samples calls for, averaged over the batch.
    /// </summary>
    /// <param name="input">
    /// Input values for each sample.
    /// </param>
    /// <param name="targets">
    /// The desired output value for each sample.
    /// </param>
    /// <returns>
    /// Returns the mean gradients for the connection weights, followed by the one for the bias.
    /// </returns>
    public (IReadOnlyList<double> WeightGradients, double BiasGradient) CalculateBatchGradients(
        IReadOnlyList<IReadOnlyList<double>> input,
        IReadOnlyList<double> targets)
    {
        input.Count.ThrowIfNotEqualTo(targets.Count);

        double[] weightGradients = new double[Weights.Length];
        double biasGradient = 0;

        for (int sampleIndex = 0; sampleIndex < input.Count; sampleIndex++)
        {
            (IReadOnlyList<double> sampleWeights, double sampleBias) =
                CalculateGradients(input[sampleIndex], targets[sampleIndex]);

            for (int index = 0; index < weightGradients.Length; index++)
                weightGradients[index] += sampleWeights[index] / input.Count;

            biasGradient += sampleBias / input.Count;
        }

        return (weightGradients, biasGradient);
    }

    #endregion
}
