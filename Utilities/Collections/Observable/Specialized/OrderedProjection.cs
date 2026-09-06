using Daylin.Utilities.Collections.Observable;
using Daylin.Utilities.Collections.Observable.Observers;
using Daylin.Utilities.Disposable;

using System.Collections.ObjectModel;

/// <summary>
/// Maintains a live, ordered projection of a read-only observable collection.
/// </summary>
/// <remarks>
/// The <see cref="Items"/> property exposes a read-only view of the ordered projection. The projection automatically
/// reflects additions and removals from the source collection without modifying it.
/// <para>
/// This class allows the order of items in a list to be managed independently of the contents of the list. For 
/// example, a single source list could manage the contents, while multiple projections could manage ordering
/// independently of each other.
/// </para>
/// </remarks>
/// <typeparam name="TItem">
/// Type of item contained in the source collection.
/// </typeparam>
/// <typeparam name="TId">
/// Type of the item identifier. Each item’s identifier must be unique within the source collection and must remain
/// stable for the item’s lifetime.
/// </typeparam>
public class OrderedProjection<TItem, TId> : DisposableObject
    where TId : notnull
{
    #region Construction

    public OrderedProjection(
        IObservableReadOnlyList<TItem> source,
        Func<TItem, TId> idSelector,
        Func<IReadOnlyList<TItem>, TItem, int>? insertionIndexSelector = null)
    {
        source.ThrowIfNull();
        idSelector.ThrowIfNull();

        Source = source;
        IdSelector = idSelector;
        InsertionIndexSelector = insertionIndexSelector;

        foreach (TItem item in Source)
        {
            TId id = IdSelector(item);

            if (!ItemsById.TryAdd(id, item))
                throw new InvalidOperationException($"Duplicate ID detected during construction: {id}");
        }

        Items = new ReadOnlyObservableCollection<TItem>(MutableItems);

        Observer = new CollectionObserver<TItem>(Source);
        Observer.ItemAdded += HandleItemAdded;
        Observer.ItemRemoved += HandleItemRemoved;
    }

    #endregion

    #region Public

    /// <summary>
    /// The ordered collection representing the projection.
    /// </summary>
    public ReadOnlyObservableCollection<TItem> Items { get; }

    /// <summary>
    /// A list of item identifiers representing the current order.
    /// </summary>
    public IReadOnlyList<TId> Order => OrderList.AsReadOnly();

    /// <summary>
    /// Sets the order of items in the projection.
    /// </summary>
    /// <param name="newOrder">
    /// Sequence of identifiers representing the desired order.
    /// </param>
    public void SetOrder(IEnumerable<TId> newOrder)
    {
        newOrder.ThrowIfNull();

        List<TId> newOrderList = newOrder.ToList();
        HashSet<TId> seen = [];

        foreach (TId id in newOrderList)
        {
            if (!seen.Add(id))
                throw new InvalidOperationException($"Order contains duplicate identifier: {id}");

            if (!ItemsById.ContainsKey(id))
                throw new InvalidOperationException($"ID in order not found in source: {id}");
        }

        OrderList.Clear();
        OrderList.AddRange(newOrderList);

        ApplyOrder();
    }

    /// <summary>
    /// Moves an item within the ordered projection.
    /// </summary>
    /// <param name="oldIndex">The item's current index.</param>
    /// <param name="newIndex">The index to move the item to.</param>
    public void Move(int oldIndex, int newIndex)
    {
        MutableItems.Move(oldIndex, newIndex);

        TId id = OrderList[oldIndex];
        OrderList.RemoveAt(oldIndex);
        OrderList.Insert(newIndex, id);
    }

    #endregion

    #region Protected

    protected override void ReleaseResources()
    {
        Observer.ItemAdded -= HandleItemAdded;
        Observer.ItemRemoved -= HandleItemRemoved;
        Observer.Dispose();

        base.ReleaseResources();
    }

    #endregion

    #region Private

    private IObservableReadOnlyList<TItem> Source { get; }

    private Func<TItem, TId> IdSelector { get; }

    private Func<IReadOnlyList<TItem>, TItem, int>? InsertionIndexSelector { get; }

    private Dictionary<TId, TItem> ItemsById { get; } = [];

    private List<TId> OrderList { get; } = [];

    private CollectionObserver<TItem> Observer { get; }

    private ObservableCollection<TItem> MutableItems { get; } = [];

    private void HandleItemAdded(object? sender, CollectionChangedEventArgs<TItem> args)
    {
        TItem item = args.Item;
        TId id = IdSelector(item);

        if (!ItemsById.TryAdd(id, item))
            throw new InvalidOperationException($"Duplicate ID detected when adding item: {id}");

        if (InsertionIndexSelector is null)
            return;

        if (OrderList.Contains(id))
            return;

        int insertIndex = InsertionIndexSelector(MutableItems, item);
        insertIndex = Math.Clamp(insertIndex, 0, MutableItems.Count);

        MutableItems.Insert(insertIndex, item);
        OrderList.Insert(insertIndex, id);
    }

    private void HandleItemRemoved(object? sender, CollectionChangedEventArgs<TItem> args)
    {
        TItem item = args.Item;
        TId id = IdSelector(item);

        ItemsById.Remove(id);

        TItem? existing = MutableItems.FirstOrDefault(x => EqualityComparer<TId>.Default.Equals(IdSelector(x), id));
        if (existing != null)
            MutableItems.Remove(existing);

        OrderList.Remove(id);
    }

    private void ApplyOrder()
    {
        Dictionary<TId, int> orderIndex = OrderList
            .Select((id, index) => new { id, index })
            .ToDictionary(x => x.id, x => x.index);

        List<TItem> sorted = MutableItems
            .OrderBy(item =>
            {
                TId id = IdSelector(item);
                return orderIndex.TryGetValue(id, out int orderPos) ? orderPos : int.MaxValue;
            })
            .ToList();

        for (int index = 0; index < sorted.Count; index++)
        {
            if (!ReferenceEquals(MutableItems[index], sorted[index]))
                MutableItems.Move(MutableItems.IndexOf(sorted[index]), index);
        }
    }

    #endregion
}
