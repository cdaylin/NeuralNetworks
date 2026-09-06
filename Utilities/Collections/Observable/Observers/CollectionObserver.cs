using Daylin.Utilities.Disposable;
using Daylin.Utilities.Events;

using System.Collections;
using System.Collections.Specialized;

namespace Daylin.Utilities.Collections.Observable.Observers;

/// <summary>
/// Monitors an observable collection and raises an event each time an item is added or removed.
/// </summary>
/// <remarks>
/// Events are raised only for changes to the contents of the collection. Changes to the order of items within the
/// collection are ignored.
/// <para>
/// When the collection is reset (i.e.: cleared), the <see cref="ItemRemoved"/> event is raised for each item that was
/// in the collection. In order support this functionality, the contents of the monitored collection are also stored
/// within the collection observer. This is inefficient, but it is necessary. The <see cref="INotifyCollectionChanged"/>
/// 'reset' event does not indicate which items have been removed.
/// </para><para>
/// For simplicity and efficiency, the collection contents are stored in a <see cref="HashSet{T}"/>. This only works
/// under the simplifying assumption that items in the monitored collection are distinct. If it becomes necessary to
/// allow equal items to be contained in a monitored collection, this class can be enhanced to track item counts.
/// </para>
/// </remarks>
/// <typeparam name="T">
/// Type of items stored in the collection.
/// </typeparam>
public class CollectionObserver<T> : DisposableObject
{
    #region Construction

    public CollectionObserver(INotifyCollectionChanged collectionChangedNotifier)
        : this(collectionChangedNotifier, (IEnumerable<T>)collectionChangedNotifier)
    {
    }

    public CollectionObserver(
        INotifyCollectionChanged collectionChangedNotifier,
        IEnumerable<T> collection)
    {
        collectionChangedNotifier.ThrowIfNull();
        collection.ThrowIfNull();

        CollectionChangedNotifier = collectionChangedNotifier;
        Collection = collection;

        ChangedHandler = new CollectionChangedHandler(this);

        CollectionChangedNotifier.CollectionChanged += ChangedHandler.HandleCollectionChanged;
    }

    #endregion

    #region Public

    public event EventHandler<CollectionChangedEventArgs<T>>? ItemAdded
    {
        add
        {
            ThrowExceptionIfDisposed();
            ItemAddedEventManager.Event += value;
        }

        remove => ItemAddedEventManager.Event -= value;
    }


    public event EventHandler<CollectionChangedEventArgs<T>>? ItemRemoved
    {
        add
        {
            ThrowExceptionIfDisposed();
            ItemRemovedEventManager.Event += value;
        }

        remove => ItemRemovedEventManager.Event -= value;
    }

    #endregion

    #region Protected

    protected override void ReleaseResources()
    {
        ItemAddedEventManager.Dispose();
        ItemRemovedEventManager.Dispose();

        CollectionChangedNotifier.CollectionChanged -= ChangedHandler.HandleCollectionChanged;

        base.ReleaseResources();
    }

    #endregion

    #region Private

    private EventManager<CollectionChangedEventArgs<T>> ItemAddedEventManager { get; } = new();

    private EventManager<CollectionChangedEventArgs<T>> ItemRemovedEventManager { get; } = new();

    private INotifyCollectionChanged CollectionChangedNotifier { get; }

    private IEnumerable<T> Collection { get; }

    private CollectionChangedHandler ChangedHandler { get; }

    private sealed class CollectionChangedHandler : CollectionChangedEventHandler
    {
        public CollectionChangedHandler(CollectionObserver<T> observer)
        {
            Observer = observer;

            AddItems(observer.Collection);
        }

        protected override void HandleAdd(INotifyCollectionChanged? sender, IList newItems, int startIndex)
        {
            AddItems(newItems.OfType<T>());
        }

        protected override void HandleRemove(INotifyCollectionChanged? sender, IList oldItems, int startIndex)
        {
            RemoveItems(oldItems.OfType<T>());
        }

        protected override void HandleReplace(
            INotifyCollectionChanged? sender,
            IList oldItems,
            IList newItems,
            int startIndex)
        {
            RemoveItems(oldItems.OfType<T>());
            AddItems(newItems.OfType<T>());
        }

        protected override void HandleReset(INotifyCollectionChanged? sender)
        {
            RemoveItems(ItemSet.ToList());
            AddItems(Observer.Collection);
        }

        private CollectionObserver<T> Observer { get; }

        /// <summary>
        /// Stores all entities in the monitored collection.
        /// </summary>
        /// <remarks>
        /// This is necessary in order to handle the "reset" event for the entity collection.  When the collection
        /// has been "reset", the previous contents of the collection are no longer available.  This set allows
        /// the previous contents to be accessed for the purpose of unregistering event handlers.
        /// </remarks>
        private HashSet<T> ItemSet { get; } = [];

        private void AddItems(IEnumerable<T>? items)
        {
            if (items is null)
                return;

            foreach (T item in items)
            {
                if (!ItemSet.Add(item))
                {
                    throw new InvalidOperationException(
                        "The collection must contain distinct items (i.e.: it must not contain" +
                            " multiple references to the same item or to items that are equal).");
                }

                Observer.ItemAddedEventManager.RaiseEvent(Observer, new CollectionChangedEventArgs<T>(item));
            }
        }

        private void RemoveItems(IEnumerable<T>? items)
        {
            if (items is null)
                return;

            foreach (T item in items)
            {
                ItemSet.Remove(item);

                Observer.ItemRemovedEventManager.RaiseEvent(Observer, new CollectionChangedEventArgs<T>(item));
            }
        }
    }

    #endregion
}
