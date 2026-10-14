using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Backpropagation.LearningRate;

/// <summary>
/// A learning rate scheduler that uses a constant learning rate.
/// </summary>
public sealed class ConstantLearningRateScheduler : ILearningRateScheduler
{
    /// <summary>
    /// Constructs a scheduler giving one rate for every epoch.
    /// </summary>
    /// <param name="learningRate">
    /// A positive value representing the learning rate.
    /// </param>
    [JsonConstructor]
    public ConstantLearningRateScheduler(float learningRate)
    {
        // Positive rather than merely not negative:  a rate of zero trains every batch, counts every epoch,
        //  advances the optimizer's state, and moves no parameter, which is a silent no-op rather than training.
        learningRate.ThrowIfNotPositive();

        LearningRate = learningRate;
    }

    /// <summary>
    /// The learning rate used for all training epochs.
    /// </summary>
    public float LearningRate { get; }

    /// <inheritdoc cref="ILearningRateScheduler.GetLearningRate(int)"/>
    public float GetLearningRate(int epochCount) => LearningRate;
}
