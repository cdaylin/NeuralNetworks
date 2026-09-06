namespace Daylin.Utilities.Extensions;

/// <summary>
/// Extension methods for arrays.
/// </summary>
public static class ArrayExtensions
{
    public static T[] Fill<T>(this T[] array, T value)
        where T : struct
    {
        array.ThrowIfNull();

        for (int index = 0; index < array.Length; index++)
            array[index] = value;

        return array;
    }

    public static T[] Clone<T>(this T[] array)
    {
        array.ThrowIfNull();

        return (T[])array.Clone();
    }
}
