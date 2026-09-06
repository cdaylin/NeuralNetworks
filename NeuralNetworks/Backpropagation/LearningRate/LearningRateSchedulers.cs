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
    /// A new learning rate scheduler that uses a constant learning rate for all training epochs.
    /// </returns>
    public static ILearningRateScheduler ConstantLearningRateScheduler(float learningRate) =>
        new ConstantLearningRateScheduler(learningRate);
}
