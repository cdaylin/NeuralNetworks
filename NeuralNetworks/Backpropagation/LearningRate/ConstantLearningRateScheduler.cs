using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Backpropagation.LearningRate;

/// <summary>
/// A learning rate scheduler that uses a constant learning rate.
/// </summary>
public sealed class ConstantLearningRateScheduler : ILearningRateScheduler
{
    [JsonConstructor]
    public ConstantLearningRateScheduler(float learningRate)
    {
        learningRate.ThrowIfNegative();

        LearningRate = learningRate;
    }

    /// <summary>
    /// The learning rate used for all training epochs.
    /// </summary>
    public float LearningRate { get; }

    public float GetLearningRate(int epochCount) => LearningRate;
}
