using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Backpropagation.LearningRate;

/// <summary>
/// A learning rate scheduler that reduces the learning rate by a constant factor each epoch.
/// </summary>
/// <remarks>
/// The learning rate for an epoch is <see cref="InitialLearningRate"/> multiplied by
/// <see cref="DecayRate"/> raised to the number of epochs completed, so the first epoch trains at the initial
/// rate and each epoch after it at a fixed fraction of the one before.
/// </remarks>
public sealed class ExponentialDecayLearningRateScheduler : ILearningRateScheduler
{
    #region Construction

    /// <summary>
    /// Constructs a scheduler.
    /// </summary>
    /// <param name="initialLearningRate">
    /// The learning rate for the first epoch.
    /// </param>
    /// <param name="decayRate">
    /// The fraction of an epoch's learning rate that the next epoch keeps.  (See <see cref="DecayRate"/>.)
    /// </param>
    [JsonConstructor]
    public ExponentialDecayLearningRateScheduler(float initialLearningRate, float decayRate)
    {
        initialLearningRate.ThrowIfNotPositive();
        decayRate.ThrowIfNotPositive();
        decayRate.ThrowIfGreaterThan(1);

        InitialLearningRate = initialLearningRate;
        DecayRate = decayRate;
    }

    #endregion

    #region Public

    /// <summary>
    /// The learning rate used for the first training epoch.
    /// </summary>
    public float InitialLearningRate { get; }

    /// <summary>
    /// The fraction of an epoch's learning rate that the next epoch keeps, in the range (0, 1].
    /// </summary>
    /// <remarks>
    /// A decay rate of one leaves the learning rate constant.
    /// </remarks>
    public float DecayRate { get; }

    /// <inheritdoc cref="ILearningRateScheduler.GetLearningRate(int)"/>
    public float GetLearningRate(int epochCount)
    {
        epochCount.ThrowIfNegative();

        return (float)(InitialLearningRate * Math.Pow(DecayRate, epochCount));
    }

    #endregion
}
