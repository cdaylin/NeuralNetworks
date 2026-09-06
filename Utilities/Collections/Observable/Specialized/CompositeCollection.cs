using Daylin.Utilities.Collections.Observable.Observers;

using System.Collections;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;

namespace Daylin.Utilities.Collections.Observable.Specialized;

/// <summary>
/// A collection that maintains a read-only view of zero-to-many collections as if they are a single collection.
/// </summary>
/// <remarks>
/// The contents of all source collections are copied into the composite collection, resulting in a read-only
/// collection that contains the contents of all source collections.  The contents are ordered according to the
/// order in which source collections are registered, with all items from one source preceding all items from the
/// subsequent source.
/// <para>
/// If a source collection implements <see cref="INotifyCollectionChanged"/>, then the composite collection will be
/// automatically updated in response to changes to the contents of the source collection.  If a source collection 
/// does not implement <see cref="INotifyCollectionChanged"/>, then changes to the contents of the source collection
/// will not be reflected in the contents of the <see cref="CompositeCollection{T}"/>.
/// </para>
/// </remarks>
/// <typeparam name="T">
/// Type of items in the collection.
/// </typeparam>
public sealed class CompositeCollection<T>
    : ReadOnlyCollection<T>, INotifyPropertyChanged, INotifyCollectionChanged, IDisposable
{
    #region Construction

    public CompositeCollection()
        : base(new ObservableCollection<T>())
    {
        CollectionChangedHandler = new SourceCollectionChangedEventHandler(this);

        RegisterForEvents();
    }

    #endregion

    #region Public

    public void Dispose()
    {
        if (IsDisposed)
            return;

        try
        {
            UnregisterForEvents();
            UnregisterForItemSourceEvents();

            // release subscribers for garbage collection
            CollectionChanged = null;
            PropertyChanged = null;

            // release aggregated objects for garbage collection
            Sources.Clear();
            SourceCounts.Clear();
        }
        finally
        {
            IsDisposed = true;
            GC.SuppressFinalize(this);
        }
    }

    public event NotifyCollectionChangedEventHandler? CollectionChanged;

    public event PropertyChangedEventHandler? PropertyChanged;


    public void AddItemSource(IEnumerable<T> source)
    {
        InsertItemSource(Sources.Count, source);
    }

    public void InsertItemSource(int index, IEnumerable<T> source)
    {
        source.ThrowIfNull();

        if (ReferenceEquals(source, this))
            throw new ArgumentException(
                $"A {nameof(CompositeCollection<T>)} may not be added to itself as a source.");

        if (index < 0 || index > Sources.Count)
            throw new ArgumentOutOfRangeException(nameof(source));

        if (Sources.Contains(source))
            throw new ArgumentException("A source may not be added more than once.", nameof(source));

        SourceCounts.Add(source, source.Count());
        Sources.Insert(index, source);

        RegisterForItemSourceEvents(source);

        Items.InsertRange(GetStartIndex(source), source);
    }

    public void RemoveItemSource(IEnumerable<T> source)
    {
        source.ThrowIfNull();

        if (!Sources.Contains(source))
            throw new ArgumentException("Item source not found.", nameof(source));

        UnregisterForItemSourceEvents(source);

        Items.RemoveRange(GetStartIndex(source), SourceCounts[source]);

        Sources.Remove(source);
        SourceCounts.Remove(source);
    }

    #endregion

    #region Private

    private bool IsDisposed { get; set; }

    /// <summary>
    /// Returns the index into <see cref="Items"/> at which items from a specified source collection are stored.
    /// </summary>
    private int GetStartIndex(IEnumerable<T> source)
    {
        int sourceIndex = Sources.IndexOf(source);

        int count = 0;

        for (int index = 0; index < sourceIndex; index++)
            count += SourceCounts[Sources[index]];

        return count;
    }

    /// <summary>
    /// Invoked when the contents have changed for <see cref="Items"/>, the wrapped collection.
    /// </summary>
    private void HandleItemsCollectionChanged(object? sender, NotifyCollectionChangedEventArgs args)
    {
        CollectionChanged?.Invoke(this, args);
    }

    /// <summary>
    /// Invoked when the value has changed for a public property on <see cref="Items"/>, the wrapped collection.
    /// </summary>
    private void HandleItemsPropertyChanged(object? sender, PropertyChangedEventArgs args)
    {
        // If this object has a public property of the same name, raise the property-changed event.

        bool hasProperty = GetType()
            .GetProperties()
            .Any(property => property.Name.Equals(args.PropertyName, StringComparison.Ordinal)
                && property.GetMethod!.IsPublic);

        if (hasProperty)
            PropertyChanged?.Invoke(this, args);
    }

    private void RegisterForEvents()
    {
        Items.CollectionChanged += HandleItemsCollectionChanged;
        ((INotifyPropertyChanged)Items).PropertyChanged += HandleItemsPropertyChanged;
    }

    private void UnregisterForEvents()
    {
        Items.CollectionChanged -= HandleItemsCollectionChanged;
        ((INotifyPropertyChanged)Items).PropertyChanged -= HandleItemsPropertyChanged;
    }

    private void RegisterForItemSourceEvents(IEnumerable<T> source)
    {
        if (source is INotifyCollectionChanged collectionChangedNotifier)
            collectionChangedNotifier.CollectionChanged += CollectionChangedHandler.HandleCollectionChanged;
    }

    private void UnregisterForItemSourceEvents(IEnumerable<T> source)
    {
        if (source is INotifyCollectionChanged collectionChangedNotifier)
            collectionChangedNotifier.CollectionChanged -= CollectionChangedHandler.HandleCollectionChanged;
    }

    private void UnregisterForItemSourceEvents()
    {
        foreach (IEnumerable<T> source in Sources)
            UnregisterForItemSourceEvents(source);
    }

    private new ObservableCollection<T> Items => (ObservableCollection<T>)base.Items;

    /// <summary>
    /// Ordered list of source collections.
    /// </summary>
    private IList<IEnumerable<T>> Sources { get; } = [];

    /// <summary>
    /// Maps each source collection to the number of items in that collection at the most recent time that
    /// this collection was updated to contain that collection's items.
    /// </summary>
    private IDictionary<IEnumerable<T>, int> SourceCounts { get; } = new Dictionary<IEnumerable<T>, int>();

    private SourceCollectionChangedEventHandler CollectionChangedHandler { get; }

    private class SourceCollectionChangedEventHandler : CollectionChangedEventHandler
    {
        public SourceCollectionChangedEventHandler(CompositeCollection<T> compositeCollection)
        {
            CompositeCollection = compositeCollection;
        }

        private CompositeCollection<T> CompositeCollection { get; }

        protected override void HandleAdd(INotifyCollectionChanged? sender, IList newItems, int startIndex)
        {
            IEnumerable<T> itemSource = (IEnumerable<T>)sender!;

            int collectionStartIndex = CompositeCollection.GetStartIndex(itemSource);

            CompositeCollection.Items.InsertRange(
                collectionStartIndex + startIndex, newItems.Cast<T>());

            CompositeCollection.SourceCounts[itemSource] += newItems.Count;
        }

        protected override void HandleRemove(INotifyCollectionChanged? sender, IList oldItems, int startIndex)
        {
            IEnumerable<T> itemSource = (IEnumerable<T>)sender!;

            int collectionStartIndex = CompositeCollection.GetStartIndex(itemSource);

            CompositeCollection.Items.RemoveRange(collectionStartIndex + startIndex, oldItems.Count);

            CompositeCollection.SourceCounts[itemSource] -= oldItems.Count;
        }

        protected override void HandleReplace(
            INotifyCollectionChanged? sender,
            IList oldItems,
            IList newItems,
            int startIndex)
        {
            IEnumerable<T> itemSource = (IEnumerable<T>)sender!;

            int collectionStartIndex = CompositeCollection.GetStartIndex(itemSource);

            CompositeCollection.Items.SetRange(
                collectionStartIndex + startIndex,
                newItems.Cast<T>().ToArray(),
                shouldSkipEqualItems: true);
        }

        protected override void HandleMove(
            INotifyCollectionChanged? sender,
            IList newItems,
            int oldStartIndex,
            int newStartIndex)
        {
            IEnumerable<T> itemSource = (IEnumerable<T>)sender!;

            int collectionStartIndex = CompositeCollection.GetStartIndex(itemSource);

            for (int itemCount = 0; itemCount < newItems.Count; itemCount++)
                CompositeCollection.Items.Move(
                    oldStartIndex + collectionStartIndex + itemCount,
                    newStartIndex + collectionStartIndex + itemCount);
        }

        protected override void HandleReset(INotifyCollectionChanged? sender)
        {
            IEnumerable<T> originalSource = (IEnumerable<T>)sender!;

            int collectionStartIndex = CompositeCollection.GetStartIndex(originalSource);

            int oldCount = CompositeCollection.SourceCounts[originalSource];

            IReadOnlyList<T> newItems =
                originalSource as IReadOnlyList<T> ?? originalSource.ToArray();

            int newCount = newItems.Count;

            if (oldCount == newCount)
                CompositeCollection.Items.SetRange(
                    collectionStartIndex,
                    newItems,
                    shouldSkipEqualItems: true);
            else
            {
                CompositeCollection.Items.RemoveRange(collectionStartIndex, oldCount);
                CompositeCollection.Items.InsertRange(collectionStartIndex, newItems);
            }

            CompositeCollection.SourceCounts[originalSource] = newCount;
        }
    }

    #endregion
}