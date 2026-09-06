using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Backpropagation.LearningRate;

/// <summary>
/// Responsible for adjusting the learning rate while training a neural network.
/// </summary>
[JsonPolymorphic]
[JsonDerivedType(typeof(ConstantLearningRateScheduler), typeDiscriminator: "Constant")]
public interface ILearningRateScheduler
{
    /// <summary>
    /// Returns the learning rate to use during the next training epoch.
    /// </summary>
    /// <param name="epochCount">
    /// The number of completed training epochs.
    /// </param>
    /// <returns>
    /// Returns a positive number representing the learning rate to use during the next epoch.
    /// </returns>
    float GetLearningRate(int epochCount);
}
