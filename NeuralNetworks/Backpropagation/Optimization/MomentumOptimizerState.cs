using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Backpropagation.Optimization;

/// <summary>
/// The velocity a <see cref="MomentumOptimizer"/> carries for a group of parameters.
/// </summary>
internal sealed class MomentumOptimizerState : IOptimizerState
{
    #region Construction

    /// <summary>
    /// Constructs the state for a group of parameters, with every velocity at zero.
    /// </summary>
    /// <param name="parameterCount">
    /// Number of parameters in the group, which may be zero.
    /// </param>
    public MomentumOptimizerState(int parameterCount)
    {
        parameterCount.ThrowIfNegative();

        Velocities = new float[parameterCount];
    }

    /// <summary>
    /// Constructs the state a saved session carried.
    /// </summary>
    /// <param name="velocities">
    /// The velocity carried for each parameter.
    /// </param>
    [JsonConstructor]
    public MomentumOptimizerState(float[] velocities)
    {
        velocities.ThrowIfNull();

        Velocities = velocities;
    }

    #endregion

    #region Public

    /// <inheritdoc cref="IOptimizerState.ParameterCount"/>
    [JsonIgnore]
    public int ParameterCount => Velocities.Length;

    /// <summary>
    /// The velocity carried for each parameter:  the gradients of earlier steps, each decayed once per step by
    /// the optimizer's momentum.
    /// </summary>
    public float[] Velocities { get; }

    #endregion
}
