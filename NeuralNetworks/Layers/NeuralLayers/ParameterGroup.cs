using Daylin.NeuralNetworks.Backpropagation.Optimization;

namespace Daylin.NeuralNetworks.Layers.NeuralLayers;

/// <summary>
/// Parameters a layer holds together, with the cost gradients accumulated for them and the state its optimizer
/// carries.
/// </summary>
/// <remarks>
/// A group is a layer's bias vector, or the connection weights of one of its output values.  Grouping them is
/// what lets an optimizer that keeps state per parameter -- a velocity, a moment estimate -- hold that state
/// in arrays the same shape as the parameters themselves.
/// <para>
/// A group holds the parameters by reference, so moving them moves the layer's own weights and biases.  A
/// convolutional layer's shared weights are one group however many output values use them, which is what makes
/// every span's gradient accumulate into the same place.
/// </para><para>
/// A group may hold no parameters at all:  a sparse layer's output value whose every connection is masked has
/// an empty group of weights, and is trained through its bias alone.
/// </para><para>
/// A batch moves the parameters in three steps, because a limit on the norm of the whole network's gradients
/// cannot be known until every group has taken its mean, and because every group's optimizer state is settled
/// before any group moves:  <see cref="TakeMeanGradients"/> averages this group's gradients and returns its
/// share of that norm, <see cref="PrepareOptimizerState"/> gives the group the state its step needs, and
/// <see cref="ApplyGradients"/> then moves the parameters by the factor the norm decided.
/// <see cref="NeuralNetwork.ApplyGradients"/> is the one caller and makes the three passes in that order.  The
/// norm comes back from the averaging rather than from a member of its own, so it cannot be read before the
/// gradients are averaged, and <see cref="ApplyGradients"/> refuses a group whose state was never settled, so
/// the order cannot be quietly departed from either.
/// </para>
/// </remarks>
internal sealed class ParameterGroup
{
    #region Construction

    /// <summary>
    /// Constructs a group over a layer's parameters, held by reference.
    /// </summary>
    /// <param name="parameters">
    /// The parameters, which the group moves in place.  An empty group is allowed.
    /// </param>
    public ParameterGroup(float[] parameters)
    {
        parameters.ThrowIfNull();

        Parameters = parameters;
        Gradients = new float[parameters.Length];
    }

    #endregion

    #region Public

    /// <summary>
    /// The parameters themselves, held by reference.
    /// </summary>
    public float[] Parameters { get; }

    /// <summary>
    /// The gradient accumulated for each parameter since the last time they moved.
    /// </summary>
    /// <remarks>
    /// <see cref="TakeMeanGradients"/> replaces each accumulated gradient with the mean over the batch's
    /// samples, so from that point in a batch until the parameters move these are mean gradients.
    /// </remarks>
    public float[] Gradients { get; }

    /// <summary>
    /// State the optimizer carries for this group, or <see langword="null"/> where it carries none.
    /// </summary>
    public IOptimizerState? OptimizerState { get; set; }

    /// <summary>
    /// Replaces each accumulated gradient with the mean over the samples it was accumulated from, and returns
    /// this group's share of the norm of the network's gradients.
    /// </summary>
    /// <remarks>
    /// Averaged in place, so that a batch divides each gradient once however the network's gradients are
    /// afterwards scaled.  The norm comes back from here rather than from a member of its own, so that it cannot
    /// be read from gradients that have not been averaged;  it is summed in <see langword="double"/>, so that a
    /// network of many parameters neither overflows nor loses the small contributions.
    /// </remarks>
    /// <param name="sampleCount">
    /// Number of samples the gradients were accumulated from.
    /// </param>
    /// <param name="measureNorm">
    /// Whether the caller will use the norm, which it will not where the network has no limit on its gradients.
    /// </param>
    /// <returns>
    /// Returns the sum of the squares of the mean gradients it wrote, or zero where
    /// <paramref name="measureNorm"/> is <see langword="false"/>.
    /// </returns>
    public double TakeMeanGradients(int sampleCount, bool measureNorm)
    {
        sampleCount.ThrowIfNotPositive();

        // A batch of one sample needs no division, and a network with no limit on its gradients needs no norm,
        //  so the common case of training sample by sample without a limit walks nothing here.  The branches are
        //  outside the loops rather than inside, since neither condition changes across a group.
        if (sampleCount != 1)
        {
            if (measureNorm)
                return TakeMeanGradientsAndMeasureNorm(sampleCount);

            for (int index = 0; index < Gradients.Length; index++)
                Gradients[index] /= sampleCount;

            return 0;
        }

        return measureNorm ? MeasureSquaredNorm() : 0;
    }

    /// <summary>
    /// Gives the group the optimizer state it needs to take a step, where what it holds will not serve.
    /// </summary>
    /// <remarks>
    /// A pass of its own, made for every group before any of them moves, because creating state allocates and a
    /// batch that failed halfway through moving the network would leave some parameters stepped and others not.
    /// <see cref="ApplyGradients"/> asks whether the state will serve and refuses a group where it will not,
    /// rather than settling it, so this is an obligation on the caller rather than a convenience.
    /// </remarks>
    /// <param name="optimizer">
    /// The optimizer that will move the parameters.
    /// </param>
    public void PrepareOptimizerState(IOptimizer optimizer)
    {
        optimizer.ThrowIfNull();

        if (!optimizer.CanApplyTo(OptimizerState, Parameters.Length))
            OptimizerState = optimizer.CreateState(Parameters.Length);
    }

    /// <summary>
    /// Moves the parameters against their mean gradients, and discards those gradients.
    /// </summary>
    /// <param name="optimizer">
    /// The optimizer moving the parameters.
    /// </param>
    /// <param name="learningRate">
    /// The learning rate for the training epoch in progress.
    /// </param>
    /// <param name="gradientScale">
    /// The factor every mean gradient is multiplied by before the optimizer sees it:  one, unless the
    /// network's gradients are being scaled down to a maximum norm.  Kept in <see langword="double"/>, since a
    /// large norm against a small maximum gives a factor no <see langword="float"/> can hold.
    /// </param>
    public void ApplyGradients(IOptimizer optimizer, float learningRate, double gradientScale)
    {
        optimizer.ThrowIfNull();

        if (gradientScale != 1)
        {
            for (int index = 0; index < Gradients.Length; index++)
                Gradients[index] = (float)(Gradients[index] * gradientScale);
        }

        // Prepared by the caller, for every group, before any of them moves.  Asked here rather than settled
        //  here, so that skipping that pass fails instead of quietly restoring the lazy creation it replaced.
        if (!optimizer.CanApplyTo(OptimizerState, Parameters.Length))
        {
            throw new InvalidOperationException(
                $"{optimizer.ShortName} cannot step this group from the state it holds.  Every group's state is "
                + $"settled by {nameof(PrepareOptimizerState)} before any group moves.");
        }

        optimizer.ApplyGradients(OptimizerState, Parameters, Gradients, learningRate);

        DiscardGradients();
    }

    /// <summary>
    /// Discards the gradients accumulated for the parameters, leaving the parameters themselves alone.
    /// </summary>
    public void DiscardGradients() => Array.Clear(Gradients);


    /// <summary>
    /// Adds to the gradient accumulated for each parameter, in the parameters' own order.
    /// </summary>
    /// <param name="gradients">
    /// The gradient of the cost with respect to each parameter, for one output value of one sample.
    /// </param>
    /// <exception cref="ArgumentException">
    /// The sequence holds a gradient for some number of parameters other than the group's own.
    /// </exception>
    public void AccumulateGradients(IEnumerable<float> gradients)
    {
        gradients.ThrowIfNull();

        int index = 0;

        foreach (float gradient in gradients)
        {
            // Counted before the write rather than after, so a sequence longer than the group is refused by
            //  name instead of running off the end of the array.  How much longer is not known at this point and
            //  is not claimed;  a shorter one would otherwise leave the rest of the gradients where they were,
            //  which reads as a batch that called for nothing rather than as a layer that handed over too few.
            if (index == Gradients.Length)
                throw GradientCountException($"more than {Gradients.Length}");

            Gradients[index++] += gradient;
        }

        if (index != Gradients.Length)
            throw GradientCountException(index.ToString());
    }

    /// <summary>
    /// Adds to the gradient accumulated for one parameter.
    /// </summary>
    /// <param name="index">
    /// Index of the parameter within the group.
    /// </param>
    /// <param name="gradient">
    /// The gradient of the cost with respect to that parameter, for one output value of one sample.
    /// </param>
    public void AccumulateGradient(int index, float gradient) => Gradients[index] += gradient;

    #endregion

    #region Private

    /// <summary>
    /// Divides each accumulated gradient by the sample count and sums the squares of the results.
    /// </summary>
    /// <param name="sampleCount">
    /// Number of samples the gradients were accumulated from, which is more than one.
    /// </param>
    /// <returns>
    /// Returns the sum of the squares of the mean gradients it wrote.
    /// </returns>
    private double TakeMeanGradientsAndMeasureNorm(int sampleCount)
    {
        double squaredNorm = 0;

        for (int index = 0; index < Gradients.Length; index++)
        {
            Gradients[index] /= sampleCount;

            squaredNorm += (double)Gradients[index] * Gradients[index];
        }

        return squaredNorm;
    }

    /// <summary>
    /// Sums the squares of the gradients as they stand.
    /// </summary>
    /// <returns>
    /// Returns the sum of their squares.
    /// </returns>
    private double MeasureSquaredNorm()
    {
        double squaredNorm = 0;

        foreach (float gradient in Gradients)
            squaredNorm += (double)gradient * gradient;

        return squaredNorm;
    }

    /// <summary>
    /// Builds the refusal for a sequence of gradients that does not match the group.
    /// </summary>
    /// <param name="count">
    /// How many gradients the sequence held, as text, since a sequence longer than the group is refused before
    /// it has been counted and the count is therefore not claimed.
    /// </param>
    /// <returns>
    /// Returns the exception, naming the argument at fault rather than a local.
    /// </returns>
    private ArgumentException GradientCountException(string count) =>
        new(
            $"A gradient is needed for each of the group's {Gradients.Length} parameters, and this sequence "
            + $"holds {count}.",
            "gradients");

    #endregion
}
