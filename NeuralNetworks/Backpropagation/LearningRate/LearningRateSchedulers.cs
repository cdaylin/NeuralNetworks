namespace Daylin.NeuralNetworks.Backpropagation.LearningRate;

/// <summary>
/// Provides access to commonly used learning rate schedulers for training neural networks.
/// </summary>
public static class LearningRateSchedulers
{
    /// <summary>
    /// Constructs a learning rate scheduler that uses a constant learning rate for all training epochs.
    /// </summary>
    /// <param name="learningRate">
    /// A positive value representing the learning rate.
    /// </param>
    /// <returns>
    /// Returns a new learning rate scheduler that uses a constant learning rate for all training epochs.
    /// </returns>
    public static ILearningRateScheduler ConstantLearningRateScheduler(float learningRate) =>
        new ConstantLearningRateScheduler(learningRate);

    /// <summary>
    /// Constructs a learning rate scheduler that reduces the learning rate by a constant factor each epoch.
    /// </summary>
    /// <param name="initialLearningRate">
    /// A positive value representing the learning rate for the first epoch.
    /// </param>
    /// <param name="decayRate">
    /// The fraction of an epoch's learning rate that the next epoch keeps, in the range (0, 1].
    /// </param>
    /// <returns>
    /// Returns a new learning rate scheduler that decays the learning rate exponentially over training epochs.
    /// </returns>
    public static ILearningRateScheduler ExponentialDecayLearningRateScheduler(
        float initialLearningRate,
        float decayRate) =>
        new ExponentialDecayLearningRateScheduler(initialLearningRate, decayRate);
}
