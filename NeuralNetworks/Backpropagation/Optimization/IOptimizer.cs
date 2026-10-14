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
/// </para><para>
/// The optimizers are the library's own:  an implementation from outside it trains correctly and cannot be
/// saved, since a session writes its optimizer and that optimizer's state by the discriminators listed here and
/// on <see cref="IOptimizerState"/>.  <see cref="Optimizers"/> is the supported surface.
/// </para>
/// </remarks>
[JsonPolymorphic]
[JsonDerivedType(typeof(GradientDescentOptimizer), typeDiscriminator: GradientDescentOptimizer.Discriminator)]
[JsonDerivedType(typeof(MomentumOptimizer), typeDiscriminator: MomentumOptimizer.Discriminator)]
[JsonDerivedType(typeof(AdamOptimizer), typeDiscriminator: AdamOptimizer.Discriminator)]
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
    /// State of another kind, or state created for a different number of parameters, cannot be continued from.
    /// Asking this is how a caller finds out before applying gradients:  a group holding state this refuses is
    /// given fresh state instead, which starts the optimizer over for that group, and
    /// <see cref="ApplyGradients"/> refuses such state outright.
    /// <para>
    /// What is compared is the state's kind and the number of parameters it describes, which is everything a
    /// state records.  Two optimizers of one kind with different hyperparameters therefore accept one another's
    /// state, and continuing from it is a warm restart under the new hyperparameters rather than an error.
    /// </para>
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
    /// <remarks>
    /// Three calls, in this order:  <see cref="CanApplyTo"/> to ask whether the state a group already holds can
    /// be continued from, <see cref="CreateState"/> for fresh state where it cannot, and this to take the step.
    /// The state passed here is state of this optimizer's kind, sized for this group, and is
    /// <see langword="null"/> only for an optimizer that keeps none;  anything else is refused.
    /// </remarks>
    /// <param name="state">
    /// State of this optimizer's kind, sized for this group of parameters, or <see langword="null"/> where it
    /// keeps none.
    /// </param>
    /// <param name="parameters">
    /// The parameters to move, in place.
    /// </param>
    /// <param name="gradients">
    /// The gradient of the cost with respect to each parameter, one for each parameter.
    /// </param>
    /// <param name="learningRate">
    /// The learning rate for the training epoch in progress.
    /// </param>
    /// <exception cref="ArgumentException">
    /// The state is not state this optimizer created for a group of this size.
    /// </exception>
    void ApplyGradients(
        IOptimizerState? state,
        float[] parameters,
        ReadOnlySpan<float> gradients,
        float learningRate);
}
