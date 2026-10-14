using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Backpropagation.Optimization;

/// <summary>
/// The moment estimates and step count an <see cref="AdamOptimizer"/> carries for a group of parameters.
/// </summary>
internal sealed class AdamOptimizerState : IOptimizerState
{
    #region Construction

    /// <summary>
    /// Constructs the state for a group of parameters, as it stands before any step.
    /// </summary>
    /// <param name="parameterCount">
    /// Number of parameters in the group, which may be zero.
    /// </param>
    public AdamOptimizerState(int parameterCount)
    {
        parameterCount.ThrowIfNegative();

        FirstMoments = new float[parameterCount];
        SecondMoments = new float[parameterCount];
    }

    /// <summary>
    /// Constructs the state a saved session carried.
    /// </summary>
    /// <param name="firstMoments">
    /// The decaying mean of the gradients seen for each parameter.
    /// </param>
    /// <param name="secondMoments">
    /// The decaying mean of the squared gradients seen for each parameter.
    /// </param>
    /// <param name="stepCount">
    /// Number of steps taken, which the bias correction is calculated from.
    /// </param>
    [JsonConstructor]
    public AdamOptimizerState(float[] firstMoments, float[] secondMoments, int stepCount)
    {
        firstMoments.ThrowIfNull();
        secondMoments.ThrowIfNull();
        secondMoments.Length.ThrowIfNotEqualTo(firstMoments.Length);

        // A negative count makes the bias-correction remainder for the second moment zero or negative, and the
        //  step divides by the square root of the corrected moment:  a document carrying one turns every
        //  parameter of the group into a value that is not a number, at the first step and for good.
        stepCount.ThrowIfNegative();

        // A second moment is a decaying mean of squares, so the library never writes a negative one, and the
        //  same square root is what a document carrying one would reach.  Checked element by element because
        //  nothing downstream looks:  the result is not a number, it reaches the parameters, and it then reaches
        //  the moments themselves on the next step.
        foreach (float secondMoment in secondMoments)
            secondMoment.ThrowIfNegative();

        FirstMoments = firstMoments;
        SecondMoments = secondMoments;
        StepCount = stepCount;
    }

    #endregion

    #region Public

    /// <inheritdoc cref="IOptimizerState.ParameterCount"/>
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
