namespace Daylin.Utilities.Comparisons;

/// <summary>
/// A comparer that sorts <c>null</c> values before non-<c>null</c> values.
/// </summary>
public sealed class NullComparer<T> : IComparer<T?>
{
    public NullComparer()
    {
    }

    public NullComparer(IComparer<T>? comparer = null)
    {
        Comparer = comparer;
    }

    public int Compare(T? left, T? right)
    {
        if (left is null && right is null)
            return 0;

        if (left is null)
            return -1;

        if (right is null)
            return 1;

        return Comparer?.Compare(left, right) ?? 0;
    }

    private IComparer<T>? Comparer { get; }
}
