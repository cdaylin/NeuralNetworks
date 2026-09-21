using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Backpropagation.Optimization;

/// <summary>
/// Responsible for moving a group of parameters against the gradients training has calculated for them.
/// </summary>
/// <remarks>
/// A gradient is the partial derivative of the cost function with respect to a parameter, averaged over the
/// samples in a training batch:  a parameter whose gradient is positive is one the cost would grow with, so
/// training moves it the other way.  The formulas published for optimization algorithms are written in these
/// terms, and so are the ones here.
/// <para>
/// The simplest optimizer subtracts the gradient scaled by the learning rate, and keeps no state.  Others
/// carry state from one step to the next, which the layer holds on the optimizer's behalf:  an optimizer is
/// therefore free of the parameters it works on, and one instance may serve a whole network.
/// </para><para>
/// An optimizer is serialized with the training session that uses it, so that a session resumes against the
/// optimizer whose state it carries rather than whichever one a caller supplies next.
/// </para>
/// </remarks>
[JsonPolymorphic]
[JsonDerivedType(typeof(GradientDescentOptimizer), typeDiscriminator: "GradientDescent")]
[JsonDerivedType(typeof(MomentumOptimizer), typeDiscriminator: "Momentum")]
[JsonDerivedType(typeof(AdamOptimizer), typeDiscriminator: "Adam")]
public interface IOptimizer
{
    /// <summary>
    /// Abbreviated name for the optimization algorithm.  (E.g.:  "Adam".)
    /// </summary>
    string ShortName { get; }

    /// <summary>
    /// Full name for the optimization algorithm.
    /// </summary>
    string LongName { get; }

    /// <summary>
    /// Creates the state this optimizer keeps for a group of parameters.
    /// </summary>
    /// <param name="parameterCount">
    /// Number of parameters in the group.
    /// </param>
    /// <returns>
    /// Returns the state to hold for the group, or <see langword="null"/> where the optimizer keeps none.
    /// </returns>
    IOptimizerState? CreateState(int parameterCount);

    /// <summary>
    /// Indicates whether this optimizer can continue from state a layer is holding.
    /// </summary>
    /// <remarks>
    /// State that another optimizer created, or that was created for a different number of parameters, cannot
    /// be continued from.  A layer given such state replaces it, which restarts the optimizer for that group.
    /// </remarks>
    /// <param name="state">
    /// The state the layer holds, which is <see langword="null"/> when it holds none.
    /// </param>
    /// <param name="parameterCount">
    /// Number of parameters in the group.
    /// </param>
    /// <returns>
    /// Returns <c>true</c> if and only if the optimizer can apply gradients using the state.
    /// </returns>
    bool CanApplyTo(IOptimizerState? state, int parameterCount);

    /// <summary>
    /// Moves a group of parameters against the gradients calculated for them, advancing the optimizer's state.
    /// </summary>
    /// <param name="state">
    /// State the optimizer created for this group of parameters, or <see langword="null"/> where it keeps
    /// none.
    /// </param>
    /// <param name="parameters">
    /// The parameters to move.
    /// </param>
    /// <param name="gradients">
    /// The gradient of the cost with respect to each parameter.
    /// </param>
    /// <param name="learningRate">
    /// The learning rate for the training epoch in progress.
    /// </param>
    void ApplyGradients(
        IOptimizerState? state,
        float[] parameters,
        IReadOnlyList<float> gradients,
        float learningRate);
}
