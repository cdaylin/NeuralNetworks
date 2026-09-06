using Daylin.Utilities.Comparers;

using System.ComponentModel;

namespace Daylin.Utilities.Collections.Observable.Specialized;

/// <summary>
/// An observable collection that maintains sort order.
/// </summary>
/// <typeparam name="T">
/// Type of items stored in the collection. May include <c>null</c> values.
/// </typeparam>
public class SortedObservableCollection<T> : ObservableList<T>
    where T : class, INotifyPropertyChanged
{
    public SortedObservableCollection()
    {
        Comparer = ComparerFactory.CreateComparer<T>(SortPropertyNames);
    }

    /// <summary>
    /// Configures the properties used for sorting and re-sorts the collection.
    /// </summary>
    /// <remarks>
    /// Sorting is always enabled. If <paramref name="propertyNames"/> is null or empty, the collection still sorts:
    /// <c>null</c> items are ordered before non-<c>null</c> items and all non-<c>null</c> items compare equal
    /// (stable order preserved).
    /// </remarks>
    /// <param name="propertyNames">
    /// Names of properties to sort by, in priority order. If not provided, only nullness is considered.
    /// </param>
    public void Sort(params IReadOnlyList<string>? propertyNames)
    {
        propertyNames ??= [];
        propertyNames.ThrowIfContainsNull();

        SortPropertyNames.Clear();
        SortPropertyNames.AddRange(propertyNames);

        Comparer = ComparerFactory.CreateComparer<T>(SortPropertyNames);

        PerformFullSort();
    }

    protected override void InsertItem(int insertionIndex, T newItem)
    {
        int actualIndex;

        if (IsReorderActive)
            actualIndex = insertionIndex;
        else
            actualIndex = GetInsertionIndex(newItem);

        base.InsertItem(actualIndex, newItem);

        if (!IsReorderActive)
            Subscribe(newItem);
    }

    protected override void RemoveItem(int removalIndex)
    {
        if (!IsReorderActive)
            Unsubscribe(this[removalIndex]);

        base.RemoveItem(removalIndex);
    }

    protected override void ClearItems()
    {
        foreach (T? existingItem in this)
            Unsubscribe(existingItem);

        base.ClearItems();
    }

    protected override void MoveItem(int oldIndex, int newIndex)
    {
        if (!IsReorderActive)
            throw new InvalidOperationException("Manual reordering is not allowed.");

        base.MoveItem(oldIndex, newIndex);
    }

    private List<string> SortPropertyNames { get; } = [];

    private IComparer<T?> Comparer { get; set; }

    private bool IsReorderActive { get; set; }

    // counts references to items in order to manage event subscriptions
    private Dictionary<T, int> SubscriptionCounts { get; } =
        new Dictionary<T, int>(ReferenceEqualityComparer<T>.Instance);

    private void Subscribe(T? item)
    {
        if (item is null)
            return;

        if (SubscriptionCounts.TryGetValue(item, out int count))
            SubscriptionCounts[item] = count + 1;
        else
        {
            item.PropertyChanged += HandleItemPropertyChanged;
            SubscriptionCounts[item] = 1;
        }
    }

    private void Unsubscribe(T? item)
    {
        if (item is null)
            return;

        if (!SubscriptionCounts.TryGetValue(item, out int count))
            return;

        if (count == 1)
        {
            item.PropertyChanged -= HandleItemPropertyChanged;
            SubscriptionCounts.Remove(item);
        }
        else
            SubscriptionCounts[item] = count - 1;
    }

    private void PerformFullSort()
    {
        // Stable-sorted snapshot, then reorder-in-place.
        List<T> sortedItems = this.OrderBy(item => item, Comparer).ToList();

        IsReorderActive = true;

        try
        {
            ReorderToMatch(sortedItems);
        }
        finally
        {
            IsReorderActive = false;
        }
    }

    private void ReorderToMatch(IReadOnlyList<T> orderedItems)
    {
        for (int targetIndex = 0; targetIndex < orderedItems.Count; targetIndex++)
        {
            T expectedItem = orderedItems[targetIndex];

            if (ReferenceEquals(this[targetIndex], expectedItem))
                continue;

            int foundIndex = FindIndexByReference(expectedItem, targetIndex + 1);

            Move(foundIndex, targetIndex);
        }

        int FindIndexByReference(T item, int startIndex)
        {
            for (int index = startIndex; index < Count; index++)
                if (ReferenceEquals(this[index], item))
                    return index;

            throw new InvalidOperationException("Reference not found.");
        }
    }

    private void HandleItemPropertyChanged(object? sender, PropertyChangedEventArgs eventArgs)
    {
        // handle null/empty property name as “all properties changed”
        if (!string.IsNullOrEmpty(eventArgs.PropertyName)
            && !SortPropertyNames.Contains(eventArgs.PropertyName))
            return;

        PerformFullSort();
    }

    private int GetInsertionIndex(T newItem)
    {
        return BinarySearch(newItem, Comparer, shouldReturnUpperBound: true);
    }

    private int BinarySearch(
        T? itemToInsert,
        IComparer<T?> comparer,
        bool shouldReturnUpperBound)
    {
        int lowIndex = 0;
        int highIndex = Count - 1;

        while (lowIndex <= highIndex)
        {
            int midIndex = (lowIndex + highIndex) / 2;
            int comparisonResult = comparer.Compare(this[midIndex], itemToInsert);

            if (comparisonResult < 0 || shouldReturnUpperBound && comparisonResult == 0)
                lowIndex = midIndex + 1;
            else
                highIndex = midIndex - 1;
        }

        return lowIndex;
    }
}
