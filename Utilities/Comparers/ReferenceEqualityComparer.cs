using System.Runtime.CompilerServices;

namespace Daylin.Utilities.Comparers;

/// <summary>
/// An equality comparer that considers two objects to be equal only if they are the same object.
/// </summary>
/// <typeparam name="T">
/// Type of objects to be compared.
/// </typeparam>
public sealed class ReferenceEqualityComparer<T> : IEqualityComparer<T>
    where T : class
{
    public static readonly ReferenceEqualityComparer<T> Instance = new();

    private ReferenceEqualityComparer() { }

    public bool Equals(T? left, T? right) => ReferenceEquals(left, right);

    public int GetHashCode(T item) => RuntimeHelpers.GetHashCode(item);
}