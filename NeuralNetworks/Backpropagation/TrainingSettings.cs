using Daylin.NeuralNetworks.Backpropagation.Cost;
using Daylin.NeuralNetworks.Backpropagation.LearningRate;
using Daylin.NeuralNetworks.Backpropagation.Optimization;

using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Backpropagation;

/// <summary>
/// Settings governing the training of a neural network on batches of samples.
/// </summary>
/// <remarks>
/// These settings govern how a network's parameters move in response to what training calls for, and how
/// quickly they move as it goes on:  the cost being minimized, the optimizer moving the parameters, the
/// schedule setting the rate, and the largest step the gradients may call for.  Training one sample, a batch
/// or an epoch all run under them.
/// <para>
/// They are serializable on their own, so a configuration can be written down, read back and applied to any
/// network, carrying nothing of the network or of a session's progress with it.  A
/// <see cref="TrainingSession"/> also records the settings it was started with and resumes on them, so that a
/// session cannot be continued against a cost function or an optimizer other than the one whose state it
/// carries.
/// </para><para>
/// How many samples a batch holds is not here:  a batch is the samples it is given, and dividing a set of
/// samples into batches is the business of the method doing the dividing.
/// </para>
/// </remarks>
public sealed class TrainingSettings
{
    #region Construction

    /// <summary>
    /// Constructs training settings.
    /// </summary>
    /// <param name="costFunction">
    /// The cost being minimized, which measures each of the network's output values against the value wanted
    /// and supplies the derivative training moves against.
    /// </param>
    /// <param name="optimizer">
    /// The optimizer moving the network's parameters in response to the gradients training calculates.
    /// </param>
    /// <param name="learningRateScheduler">
    /// The scheduler supplying the learning rate for each training epoch.
    /// </param>
    /// <param name="maxGradientNorm">
    /// The largest the norm of a batch's gradients may be before they are scaled down to it, or
    /// <see langword="null"/> for gradients never to be scaled.  Must be positive.
    /// </param>
    [JsonConstructor]
    public TrainingSettings(
        ICostFunction costFunction,
        IOptimizer optimizer,
        ILearningRateScheduler learningRateScheduler,
        float? maxGradientNorm = null)
    {
        costFunction.ThrowIfNull();
        optimizer.ThrowIfNull();
        learningRateScheduler.ThrowIfNull();
        maxGradientNorm?.ThrowIfNotPositive();

        CostFunction = costFunction;
        Optimizer = optimizer;
        LearningRateScheduler = learningRateScheduler;
        MaxGradientNorm = maxGradientNorm;
    }

    #endregion

    #region Public

    /// <summary>
    /// The cost being minimized, which training measures each output value with and moves against the derivative
    /// of.
    /// </summary>
    public ICostFunction CostFunction { get; }

    /// <summary>
    /// The optimizer moving the network's parameters in response to the gradients training calculates.
    /// </summary>
    public IOptimizer Optimizer { get; }

    /// <summary>
    /// The scheduler supplying the learning rate for each training epoch.
    /// </summary>
    public ILearningRateScheduler LearningRateScheduler { get; }

    /// <summary>
    /// The largest the norm of a batch's gradients may be, or <see langword="null"/> where gradients are never
    /// scaled.
    /// </summary>
    /// <remarks>
    /// The norm is the Euclidean norm of the batch's mean gradients over every parameter of the network
    /// together:  the square root of the sum of their squares, a parameter shared by several output values
    /// counted once.  Where it exceeds this maximum, every gradient is multiplied by the maximum over the norm
    /// before the optimizer sees any of them, so the gradients keep their direction and take this norm.  Where
    /// it does not, the gradients are left as they are.
    /// <para>
    /// The norm is taken over the whole network rather than layer by layer, which is what keeps the gradients'
    /// direction:  scaling each layer to its own limit would move some layers further than others relative to
    /// what the cost called for.
    /// </para>
    /// </remarks>
    public float? MaxGradientNorm { get; }

    #endregion
}
