using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Backpropagation.Optimization;

/// <summary>
/// The moment estimates and step count an <see cref="AdamOptimizer"/> carries for a group of parameters.
/// </summary>
internal sealed class AdamOptimizerState : IOptimizerState
{
    #region Construction

    public AdamOptimizerState(int parameterCount)
    {
        parameterCount.ThrowIfNotPositive();

        FirstMoments = new float[parameterCount];
        SecondMoments = new float[parameterCount];
    }

    [JsonConstructor]
    public AdamOptimizerState(float[] firstMoments, float[] secondMoments, int stepCount)
    {
        firstMoments.ThrowIfNull();
        secondMoments.ThrowIfNull();
        secondMoments.Length.ThrowIfNotEqualTo(firstMoments.Length);

        FirstMoments = firstMoments;
        SecondMoments = secondMoments;
        StepCount = stepCount;
    }

    #endregion

    #region Public

    [JsonIgnore]
    public int ParameterCount => FirstMoments.Length;

    /// <summary>
    /// The decaying mean of the gradients seen for each parameter.
    /// </summary>
    public float[] FirstMoments { get; }

    /// <summary>
    /// The decaying mean of the squared gradients seen for each parameter.
    /// </summary>
    public float[] SecondMoments { get; }

    /// <summary>
    /// Number of steps applied to this group of parameters.
    /// </summary>
    /// <remarks>
    /// Both moments start at zero and are biased towards it for as long as few steps have been taken, so the
    /// correction applied to them depends on how many there have been.
    /// </remarks>
    public int StepCount { get; set; }

    #endregion
}
