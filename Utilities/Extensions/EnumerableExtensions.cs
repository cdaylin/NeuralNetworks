using System.Diagnostics.CodeAnalysis;
using System.Numerics;

namespace Daylin.Utilities.Extensions;

/// <summary>
/// Extension methods for type <see cref="IEnumerable{T}"/>.
/// </summary>
public static class EnumerableExtensions
{
    /// <summary>
    /// Determines whether an enumerable object is <c>null</c> or empty.
    /// </summary>
    /// <typeparam name="T">
    /// Type of object in the sequence.
    /// </typeparam>
    /// <param name="enumerable">
    /// An enumerable object.
    /// </param>
    /// <returns>
    /// Returns <c>true</c> if and only if <paramref name="enumerable"/> is <c>null</c> or empty.
    /// </returns>
    public static bool IsNullOrEmpty<T>([NotNullWhen(false)] this IEnumerable<T>? enumerable)
    {
        return enumerable is null || !enumerable.Any();
    }

    /// <summary>
    /// Determines whether an enumerable contains one or more <c>null</c> values.
    /// </summary>
    /// <typeparam name="T">
    /// Type of object in the sequence.
    /// </typeparam>
    /// <param name="enumerable">
    /// A sequence.
    /// </param>
    /// <returns>
    /// Returns <c>true</c> if and only if <paramref name="enumerable"/> contains one or more <c>null</c> values.
    /// </returns>
    public static bool ContainsNull<T>(this IEnumerable<T> enumerable)
    {
        enumerable.ThrowIfNull();

        return enumerable.Any(value => value is null);
    }

    /// <summary>
    /// Returns the first element of a sequence that satisfies a condition, or returns <c>null</c> if no such
    /// element is found.
    /// </summary>
    /// <typeparam name="T">
    /// Type of object in the sequence.
    /// </typeparam>
    /// <param name="enumerable">
    /// A sequence of non-nullable value-type objects.
    /// </param>
    /// <param name="predicate">
    /// A function to test each element for a condition.
    /// </param>
    /// <returns>
    /// Returns the first element in <paramref name="enumerable"/> for which <paramref name="predicate"/> returns
    /// <c>true</c>.  Returns <c>null</c> when <paramref name="predicate"/> does not return <c>true</c> for any
    /// element in <paramref name="enumerable"/>.
    /// </returns>
    public static T? FirstOrNull<T>(this IEnumerable<T> enumerable, Func<T, bool> predicate)
        where T : struct
    {
        enumerable.ThrowIfNull();

        foreach (T item in enumerable)
        {
            if (predicate(item))
                return item;
        }

        return null;
    }

    /// <summary>
    /// Enumerates non-<c>null</c> members of a sequence.
    /// </summary>
    /// <typeparam name="T">
    /// Type of object in the sequence.
    /// </typeparam>
    /// <param name="enumerable">
    /// A sequence.
    /// </param>
    /// <returns>
    /// Enumerates those members of <paramref name="enumerable"/> that are not <c>null</c>.
    /// </returns>
    public static IEnumerable<T> WhereNotNull<T>(this IEnumerable<T?> enumerable)
    {
        enumerable.ThrowIfNull();

        foreach (T? item in enumerable)
        {
            if (item is not null)
                yield return item;
        }
    }

    /// <summary>
    /// Finds the location of an item within a sequence.
    /// </summary>
    /// <typeparam name="T">
    /// Type of object in the sequence.
    /// </typeparam>
    /// <param name="enumerable">
    /// A sequence.
    /// </param>
    /// <param name="item">
    /// Item to find.
    /// </param>
    /// <param name="comparer">
    /// Equality comparer to use for comparing the item to elements in the sequence.  If <c>null</c>, the default
    /// equality comparer for type <typeparamref name="T"/> is used.
    /// </param>
    /// <returns>
    /// Returns the index of the item, if found.  Returns <c>-1</c> if the sequence does not contain the item.
    /// </returns>
    public static int IndexOf<T>(this IEnumerable<T> enumerable, T item, IEqualityComparer<T>? comparer = null)
    {
        enumerable.ThrowIfNull();

        comparer ??= EqualityComparer<T>.Default;

        int index = 0;

        foreach (T element in enumerable)
        {
            if (comparer.Equals(item, element))
                return index;

            index += 1;
        }

        return -1;
    }

    /// <summary>
    /// Calculates the sum of a sequence of numbers.
    /// </summary>
    /// <typeparam name="T">
    /// Type of number in the sequence.
    /// </typeparam>
    /// <param name="enumerable">
    /// A sequence of numbers.
    /// </param>
    /// <returns>
    /// Returns the sum of the enumerated numbers.
    /// </returns>
    public static T Sum<T>(this IEnumerable<T> enumerable)
        where T : INumber<T>
    {
        enumerable.ThrowIfNull();

        return enumerable.Aggregate(T.Zero, (sum, value) => sum + value);
    }

    public static IEnumerable<T> ElementwiseSum<T>(this IEnumerable<T> first, IEnumerable<T> second)
        where T : INumber<T>
    {
        return first.Zip(second, (first, second) => first + second);
    }

    public static IEnumerable<T> ScalarProduct<T>(this IEnumerable<T> enumerable, T scalar)
        where T : INumber<T>
    {
        return enumerable.Select(value => value * scalar);
    }

    public static IEnumerable<T> ElementwiseProduct<T>(this IEnumerable<T> first, IEnumerable<T> second)
        where T : INumber<T>
    {
        return first.Zip(second, (first, second) => first * second);
    }

    /// <summary>
    /// Determines whether two sequences contain equal items in equal quantities, without requiring that
    /// the order of the items is the same.
    /// </summary>
    /// <typeparam name="T">
    /// Type of object in the sequences.
    /// </typeparam>
    /// <param name="left">
    /// A sequence.
    /// </param>
    /// <param name="right">
    /// A sequence.
    /// </param>
    /// <returns>
    /// Returns <c>true</c> if and only if <paramref name="left"/> and <paramref name="right"/> contain equal
    /// items in equal quantities, regardless of whether the items are ordered the same.
    /// </returns>
    public static bool AreContentsEqual<T>(this IEnumerable<T> left, IEnumerable<T> right)
    {
        left.ThrowIfNull();
        right.ThrowIfNull();

        if (left.Count() != right.Count())
            return false;

        IEnumerable<(T, int)> leftCounts = left.GetCountsByItem();
        IEnumerable<(T, int)> rightCounts = right.GetCountsByItem();

        return new HashSet<(T, int)>(leftCounts).SetEquals(rightCounts);
    }

    public static bool AreDistinct<T>(this IEnumerable<T> enumerable)
    {
        HashSet<T> values = [];

        return enumerable.All(values.Add);
    }

    /// <summary>
    /// Counts the number of times each distinct value occurs within a sequence.
    /// </summary>
    /// <typeparam name="T">
    /// Type of object in the sequence.
    /// </typeparam>
    /// <param name="enumerable">
    /// A sequence.
    /// </param>
    /// <param name="equalityComparer">
    /// The comparer to use for identifying equal items in <paramref name="enumerable"/>.  When <c>null</c>, the
    /// default comparer (<see cref="EqualityComparer{T}.Default"/> is used.
    /// </param>
    /// <returns>
    /// Returns a sequence containing one tuple per distinct item, with each tuple containing the item
    /// and the number of times the item occurred in <paramref name="enumerable"/>.
    /// </returns>
    public static IEnumerable<(T item, int count)> GetCountsByItem<T>(
        this IEnumerable<T> enumerable,
        IEqualityComparer<T>? equalityComparer = null)
    {
        equalityComparer ??= EqualityComparer<T>.Default;

        return enumerable
            .GroupBy(item => item, equalityComparer)
            .Select(group => (group.Key, group.Count()));
    }

    /// <summary>
    /// Enumerates a sequence, providing access to both the current and subsequent item at each iteration.
    /// </summary>
    /// <typeparam name="T">
    /// Type of object in the sequence.
    /// </typeparam>
    /// <param name="enumerable">
    /// A sequence.
    /// </param>
    /// <returns>
    /// Returns a tuple for each item in the sequence, each tuple containing both the current item and the
    /// next item in the sequence.  The final tuple contains the last item in the sequence and the default
    /// value for <typeparamref name="T"/>.
    /// </returns>
    public static IEnumerable<(T current, T? next)> Peek<T>(this IEnumerable<T> enumerable)
    {
        using IEnumerator<T> first = enumerable.GetEnumerator();
        using IEnumerator<T> second = enumerable.Skip(1).GetEnumerator();

        while (first.MoveNext())
        {
            T? secondValue = second.MoveNext() ? second.Current : default;

            yield return (first.Current, secondValue);
        }
    }

    /// <summary>
    /// Assembles all two-item combinations from a sequence of items.
    /// </summary>
    /// <remarks>
    /// Each item in the sequence is paired with each other item in the sequence exactly once.  Items are not
    /// paired with themselves.  Items in the sequence are not required to be distinct, however.  Duplicate items
    /// in the sequence will result in duplicate combinations and combinations containing duplicates of the same
    /// item.
    /// </remarks>
    /// <typeparam name="T">
    /// Item type.
    /// </typeparam>
    /// <param name="enumerable">
    /// A sequence.
    /// </param>
    /// <returns>
    /// Returns a sequence of tuples representing all two-item combinations created from items in the sequence.
    /// </returns>
    public static IEnumerable<(T left, T right)> GetPairs<T>(
        this IEnumerable<T> enumerable)
    {
        enumerable.ThrowIfNull();

        int count = 0;

        foreach (T left in enumerable.SkipLast(1))
        {
            count += 1;

            foreach (T right in enumerable.Skip(count))
                yield return new(left, right);
        }
    }
}
