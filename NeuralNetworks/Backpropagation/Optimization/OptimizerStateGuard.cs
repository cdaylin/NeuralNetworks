namespace Daylin.NeuralNetworks.Backpropagation.Optimization;

/// <summary>
/// The checks every optimizer makes of what it is handed, before it applies its own formula.
/// </summary>
/// <remarks>
/// Every optimizer asks the same things of a step -- are the parameters and their gradients the same number, and
/// is this state mine and the right size -- and the answers are here rather than in each optimizer, where a test
/// and the cast that depends on it would sit in different members of different files.  Named for what it does
/// rather than for what it holds, since the library's <c>...s</c> classes are factories and this creates nothing.
/// </remarks>
internal static class OptimizerStateGuard
{
    #region Internal Static

    /// <summary>
    /// Indicates whether a state is one an optimizer of this kind created for a group of this size.
    /// </summary>
    /// <typeparam name="TState">
    /// The kind of state the optimizer carries.
    /// </typeparam>
    /// <param name="state">
    /// The state a group holds, which is <see langword="null"/> when it holds none.
    /// </param>
    /// <param name="parameterCount">
    /// Number of parameters in the group.
    /// </param>
    /// <returns>
    /// Returns <c>true</c> if and only if the state can be continued from.  Two optimizers of one kind with
    /// different hyperparameters accept one another's state, since a state records none of them.
    /// </returns>
    public static bool IsStateFor<TState>(IOptimizerState? state, int parameterCount)
        where TState : class, IOptimizerState
    {
        return state is TState && state.ParameterCount == parameterCount;
    }

    /// <summary>
    /// Refuses a step whose gradients do not match its parameters.
    /// </summary>
    /// <param name="parameters">
    /// The parameters to be moved.
    /// </param>
    /// <param name="gradients">
    /// The gradient accumulated for each parameter.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// The counts differ.
    /// </exception>
    public static void RequireOneGradientPerParameter(float[] parameters, ReadOnlySpan<float> gradients)
    {
        parameters.ThrowIfNull();
        gradients.Length.ThrowIfNotEqualTo(parameters.Length);
    }

    /// <summary>
    /// Returns a state as the kind an optimizer carries, refusing anything else.
    /// </summary>
    /// <typeparam name="TState">
    /// The kind of state the optimizer carries.
    /// </typeparam>
    /// <param name="state">
    /// The state the caller supplied.
    /// </param>
    /// <param name="parameterCount">
    /// Number of parameters the state must describe.
    /// </param>
    /// <param name="optimizerName">
    /// Short name of the optimizer, for the message.
    /// </param>
    /// <returns>
    /// Returns the state, as the kind the optimizer carries.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// The state is not state of this kind, sized for a group of this size.
    /// </exception>
    public static TState Require<TState>(IOptimizerState? state, int parameterCount, string optimizerName)
        where TState : class, IOptimizerState
    {
        if (IsStateFor<TState>(state, parameterCount))
            return (TState)state!;

        throw new ArgumentException(
            $"{optimizerName} moves parameters against state of its own kind, sized for the group it is "
            + $"stepping.  {Advice}  {DescribeState(state)}, where the group holds {parameterCount}.",
            nameof(state));
    }

    /// <summary>
    /// Refuses any state at all, for an optimizer that carries none.
    /// </summary>
    /// <param name="state">
    /// The state the caller supplied.
    /// </param>
    /// <param name="optimizerName">
    /// Short name of the optimizer, for the message.
    /// </param>
    /// <exception cref="ArgumentException">
    /// The state is not <see langword="null"/>.
    /// </exception>
    public static void RequireNone(IOptimizerState? state, string optimizerName)
    {
        if (state is null)
            return;

        throw new ArgumentException(
            $"{optimizerName} carries no state, so it has nothing to continue from and ignoring what it is "
            + $"handed would leave a caller believing otherwise.  {Advice}  {DescribeState(state)}.",
            nameof(state));
    }

    #endregion

    #region Private Static

    /// <summary>
    /// The sentence telling a caller how to obtain state an optimizer accepts.
    /// </summary>
    private static string Advice =>
        $"Ask {nameof(IOptimizer.CanApplyTo)} first, and {nameof(IOptimizer.CreateState)} for state it accepts.";

    /// <summary>
    /// Describes the state a caller supplied, for a refusal.
    /// </summary>
    /// <param name="state">
    /// The state, which may be <see langword="null"/>.
    /// </param>
    /// <returns>
    /// Returns the description, which says that there is no state rather than describing one of no parameters.
    /// </returns>
    private static string DescribeState(IOptimizerState? state) =>
        state is null
            ? "No state was supplied"
            : $"This state is '{state.GetType().Name}' for {state.ParameterCount} parameters";

    #endregion
}
