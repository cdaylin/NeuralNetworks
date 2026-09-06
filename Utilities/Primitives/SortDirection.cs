namespace Daylin.Utilities.Primitives;

public enum SortDirection
{
    Ascending,
    Descending
}

public static class SortDirectionExtensions
{
    public static bool IsAscending(this SortDirection direction)
    {
        return Equals(direction, SortDirection.Ascending);
    }

    public static bool IsDescending(this SortDirection direction)
    {
        return Equals(direction, SortDirection.Descending);
    }
}
