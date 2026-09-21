using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Backpropagation.Optimization;

/// <summary>
/// The velocity a <see cref="MomentumOptimizer"/> carries for a group of parameters.
/// </summary>
internal sealed class MomentumOptimizerState : IOptimizerState
{
    #region Construction

    public MomentumOptimizerState(int parameterCount)
    {
        parameterCount.ThrowIfNotPositive();

        Velocities = new float[parameterCount];
    }

    [JsonConstructor]
    public MomentumOptimizerState(float[] velocities)
    {
        velocities.ThrowIfNull();

        Velocities = velocities;
    }

    #endregion

    #region Public

    [JsonIgnore]
    public int ParameterCount => Velocities.Length;

    /// <summary>
    /// The velocity carried for each parameter:  the gradients of earlier steps, each decayed once per step by
    /// the optimizer's momentum.
    /// </summary>
    public float[] Velocities { get; }

    #endregion
}
