using Daylin.Utilities.Primitives;

namespace Daylin.Utilities.Extensions;

/// <summary>
/// Extension methods for type <see cref="IList{T}"/>.
/// </summary>
public static class ListExtensions
{
    #region Public

    public static IList<T> AddRange<T>(this IList<T> target, IEnumerable<T> source)
    {
        target.ThrowIfNull();
        source.ThrowIfNull();

        if (target is List<T> list)
        {
            list.AddRange(source);
            return target;
        }

        foreach (T item in source)
            target.Add(item);

        return target;
    }

    public static IList<T> InsertRange<T>(this IList<T> target, int index, IEnumerable<T> source)
    {
        target.ThrowIfNull();
        source.ThrowIfNull();
        index.ThrowIfNegative();
        index.ThrowIfGreaterThan(target.Count);

        if (target is List<T> list)
        {
            list.InsertRange(index, source);
            return target;
        }

        int insertionIndex = index;

        foreach (T item in source)
            target.Insert(insertionIndex++, item);

        return target;
    }

    public static IList<T> SetRange<T>(
        this IList<T> target,
        int index,
        IReadOnlyCollection<T> source,
        bool shouldSkipEqualItems = false,
        EqualityComparer<T>? equalityComparer = null)
    {
        target.ThrowIfNull();
        source.ThrowIfNull();
        index.ThrowIfNegative();
        index.ThrowIfGreaterThan(target.Count);
        (index + source.Count).ThrowIfGreaterThan(target.Count,
            argumentExpression: "The end of the range denoted by '{nameof(index)}' and '{nameof(source)}.{nameof(Count)}'",
            comparandExpression: "the size of the target list ({target.Count}).");

        if (shouldSkipEqualItems)
            equalityComparer ??= EqualityComparer<T>.Default;

        int currentIndex = index;

        foreach (T item in source)
        {
            if (!shouldSkipEqualItems || !equalityComparer!.Equals(target[currentIndex], item))
                target[currentIndex] = item;

            currentIndex++;
        }

        return target;
    }

    public static IList<T> RemoveRange<T>(this IList<T> target, int index, int count)
    {
        target.ThrowIfNull();
        index.ThrowIfNegative();
        count.ThrowIfNegative(nameof(count));
        index.ThrowIfGreaterThan(target.Count);
        (index + count).ThrowIfGreaterThan(target.Count,
            argumentExpression: "The end of the range denoted by '{nameof(index)}' and '{nameof(count)}'",
            comparandExpression: "the size of the target list ({target.Count}).");

        if (target is List<T> list)
        {
            list.RemoveRange(index, count);
            return target;
        }

        for (int removalIndex = index + count - 1; removalIndex >= index; removalIndex--)
            target.RemoveAt(removalIndex);

        return target;
    }

    public static IList<T> InsertInSortOrder<T>(
        this IList<T> list,
        T item,
        SortDirection sortDirection = SortDirection.Ascending)
        where T : notnull, IComparable<T>
    {
        list.ThrowIfNull();

        list.InsertInSortOrder(item, (left, right) => left.CompareTo(right), sortDirection);

        return list;
    }

    public static IList<T> InsertInSortOrder<T>(
        this IList<T> list,
        T item,
        IComparer<T> comparer,
        SortDirection sortDirection = SortDirection.Ascending)
    {
        list.ThrowIfNull();
        comparer.ThrowIfNull();

        list.InsertInSortOrder(item, comparer.Compare, sortDirection);

        return list;
    }

    public static IList<T> InsertInSortOrder<T>(
        this IList<T> list,
        T item,
        Func<T, T, int> comparisonFunction,
        SortDirection sortDirection = SortDirection.Ascending)
    {
        list.ThrowIfNull();
        comparisonFunction.ThrowIfNull();

        list.Insert(list.GetSortOrderInsertIndex(item, comparisonFunction, sortDirection), item);

        return list;
    }

    #endregion

    #region Private

    private static int GetSortOrderInsertIndex<T>(
        this IList<T> list,
        T item,
        Func<T, T, int> comparisonFunction,
        SortDirection sortDirection)
    {
        int index;

        for (index = 0; index < list.Count; index++)
        {
            int comparison = comparisonFunction(item, list[index]);

            comparison = sortDirection.IsAscending() ? comparison : -comparison;

            if (comparison < 1)
                break;
        }

        return index;
    }

    #endregion
}
