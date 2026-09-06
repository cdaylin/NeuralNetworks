namespace Daylin.Utilities.Extensions;

/// <summary>
/// Extension methods for type <see cref="int"/>.
/// </summary>
public static class Int32Extensions
{
    public static bool IsEven(this int value) => value % 2 == 0;

    public static bool IsOdd(this int value) => !value.IsEven();
}
