namespace Daylin.Utilities.Mathematics;

/// <summary>
/// Performs linear interpolation (or extrapolation).
/// </summary>
public class LinearInterpolator
{
    #region Public Static

    public static decimal Interpolate(
        (decimal Input, decimal Output) firstDataPoint,
        (decimal Input, decimal Output) secondDataPoint,
        decimal input)
    {
        secondDataPoint.Input.ThrowIfEquals(firstDataPoint.Input);

        return (firstDataPoint.Output * (secondDataPoint.Input - input)
            + secondDataPoint.Output * (input - firstDataPoint.Input))
            / (secondDataPoint.Input - firstDataPoint.Input);
    }

    #endregion

    #region Construction

    /// <summary>
    /// Constructs a <see cref="LinearInterpolator"/> object.
    /// </summary>
    /// <param name="dataPoints">
    /// Known mappings of input values to output values.  At least two data points are required.  Input values
    /// must be distinct.
    /// </param>
    public LinearInterpolator(params IEnumerable<(decimal Input, decimal Output)> dataPoints)
    {
        // copy to new list, ordered by input values
        DataPoints = dataPoints?.OrderBy(dataPoint => dataPoint.Input).ToList()!;

        // validate
        DataPoints.ThrowIfNull(nameof(dataPoints));
        DataPoints.ThrowIfContainsNull(nameof(dataPoints));
        DataPoints.Count.ThrowIfLessThan(2, nameof(dataPoints));
        DataPoints.ThrowIfNotDistinct(nameof(dataPoints));
    }

    #endregion

    #region Public

    public decimal Interpolate(decimal input)
    {
        int index;

        for (index = 0; index < DataPoints.Count; index++)
        {
            if (DataPoints[index].Input == input)
                return DataPoints[index].Output;

            if (DataPoints[index].Input > input)
            {
                index -= 1;
                break;
            }
        }

        if (index < 0)
            index = 0;

        if (index > DataPoints.Count - 2)
            index = DataPoints.Count - 2;

        return Interpolate(
            (DataPoints[index].Input, DataPoints[index].Output),
            (DataPoints[index + 1].Input, DataPoints[index + 1].Output),
            input);
    }

    #endregion

    #region Private

    private IReadOnlyList<(decimal Input, decimal Output)> DataPoints { get; }

    #endregion
}