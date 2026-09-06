using Daylin.Utilities.Collections.Observable.Observers;
using Daylin.Utilities.Disposable;
using Daylin.Utilities.Threading;

using System.Collections;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;

namespace Daylin.Utilities.Collections.Observable.Specialized;

/// <summary>
/// Synchronizes the contents of two observable collections bidirectionally, with type projection.
/// </summary>
/// <remarks>
/// If source items implement <see cref="INotifyPropertyChanged"/>, the synchronizer subscribes to property changes
/// and updates the corresponding target item when properties change. However, multiple references to the same
/// <see cref="INotifyPropertyChanged"/> object within the source collection may not synchronize correctly, as the
/// synchronizer tracks subscriptions by reference and avoids duplicate subscriptions.
/// <para>
/// This class does not support reentrancy. Responding to a collection change by immediately making additional
/// changes to either collection (e.g., sorting, filtering) may break synchronization. Changes made during a change
/// handler are ignored to prevent infinite loops, but this may leave the collections in an inconsistent state.
/// </para>
/// </remarks>
/// <typeparam name="TSource">
/// The element type of the source collection.
/// </typeparam>
/// <typeparam name="TTarget">
/// The element type of the target collection.
/// </typeparam>
public sealed class CollectionSynchronizer<TSource, TTarget> : DisposableObject
{
    #region Construction

    public CollectionSynchronizer(
        ObservableCollection<TSource> source,
        ObservableCollection<TTarget> target,
        Func<TSource, TTarget> projectToTargetFunc,
        Func<TTarget, TSource> projectToSourceFunc,
        Func<TSource, TTarget, bool> areEquivalentFunc)
        : this(source, source, target, target, projectToTargetFunc, projectToSourceFunc, areEquivalentFunc)
    {
    }

    public CollectionSynchronizer(
        IList<TSource> sourceList,
        INotifyCollectionChanged sourceChangeNotifier,
        IList<TTarget> targetList,
        INotifyCollectionChanged targetChangeNotifier,
        Func<TSource, TTarget> projectToTargetFunc,
        Func<TTarget, TSource> projectToSourceFunc,
        Func<TSource, TTarget, bool> areEquivalentFunc)
    {
        sourceList.ThrowIfNull();
        sourceChangeNotifier.ThrowIfNull();
        targetList.ThrowIfNull();
        targetChangeNotifier.ThrowIfNull();
        projectToTargetFunc.ThrowIfNull();
        projectToSourceFunc.ThrowIfNull();
        areEquivalentFunc.ThrowIfNull();

        InitializeTargetList(sourceList, targetList, projectToTargetFunc);

        SourceChangeHandler = new ChangeHandler<TSource, TTarget>(
            sourceList,
            sourceChangeNotifier,
            targetList,
            projectToTargetFunc,
            areEquivalentFunc,
            SyncReentrancyGate);

        TargetChangeHandler = new ChangeHandler<TTarget, TSource>(
            targetList,
            targetChangeNotifier,
            sourceList,
            projectToSourceFunc,
            (targetItem, sourceItem) => areEquivalentFunc(sourceItem, targetItem),
            SyncReentrancyGate);
    }

    #endregion

    #region Protected

    protected override void ReleaseResources()
    {
        SourceChangeHandler.Dispose();
        TargetChangeHandler.Dispose();

        base.ReleaseResources();
    }

    #endregion

    #region Private

    private ChangeHandler<TSource, TTarget> SourceChangeHandler { get; }

    private ChangeHandler<TTarget, TSource> TargetChangeHandler { get; }

    private ScopedCounter SyncReentrancyGate { get; } = new();

    private void InitializeTargetList(
        IList<TSource> sourceList,
        IList<TTarget> targetList,
        Func<TSource, TTarget> projectionFunc)
    {
        targetList.Clear();

        foreach (TSource item in sourceList)
            targetList.Add(projectionFunc(item));
    }

    private sealed class ChangeHandler<TSourceItem, TTargetItem> : CollectionChangedEventHandler, IDisposable
    {
        #region Construction

        public ChangeHandler(
            IList<TSourceItem> sourceList,
            INotifyCollectionChanged sourceChangeNotifier,
            IList<TTargetItem> targetList,
            Func<TSourceItem, TTargetItem> projectionFunc,
            Func<TSourceItem, TTargetItem, bool> areEquivalentFunc,
            ScopedCounter reentrancyGate)
        {
            sourceList.ThrowIfNull();
            sourceChangeNotifier.ThrowIfNull();
            targetList.ThrowIfNull();
            projectionFunc.ThrowIfNull();
            areEquivalentFunc.ThrowIfNull();

            SourceList = sourceList;
            SourceChangeNotifier = sourceChangeNotifier;
            TargetList = targetList;
            ProjectionFunc = projectionFunc;
            AreEquivalentFunc = areEquivalentFunc;
            ReentrancyGate = reentrancyGate;

            ItemSubscriptionManager = new SubscriptionManager(HandleItemPropertyChanged);

            SourceChangeNotifier.CollectionChanged += HandleCollectionChanged;

            foreach (TSourceItem item in SourceList)
                ItemSubscriptionManager.Subscribe(item);
        }

        #endregion

        #region Public

        public void Dispose()
        {
            SourceChangeNotifier.CollectionChanged -= HandleCollectionChanged;
            ItemSubscriptionManager.UnsubscribeAll();
        }

        #endregion

        #region Protected

        protected override void HandleAdd(
            INotifyCollectionChanged? sender,
            IList newItems,
            int startIndex)
        {
            if (IsReentrant)
            {
                SubscribeNewItems(newItems);
                return;
            }

            using (BlockReentrancy())
            {
                foreach (TSourceItem item in newItems)
                {
                    TargetList.Insert(startIndex++, ProjectionFunc(item));
                    ItemSubscriptionManager.Subscribe(item);
                }
            }
        }

        protected override void HandleRemove(
            INotifyCollectionChanged? sender,
            IList oldItems,
            int startIndex)
        {
            if (IsReentrant)
            {
                UnsubscribeOldItems(oldItems);
                return;
            }

            using (BlockReentrancy())
            {
                for (int index = 0; index < oldItems.Count; index++)
                {
                    ItemSubscriptionManager.Unsubscribe((TSourceItem)oldItems[index]!);
                    TargetList.RemoveAt(startIndex);
                }
            }
        }

        protected override void HandleReplace(
            INotifyCollectionChanged? sender,
            IList oldItems,
            IList newItems,
            int startIndex)
        {
            if (IsReentrant)
            {
                ReplaceSubscriptions(oldItems, newItems);
                return;
            }

            using (BlockReentrancy())
            {
                for (int index = 0; index < newItems.Count; index++)
                {
                    int targetIndex = startIndex + index;

                    TSourceItem sourceItem = (TSourceItem)newItems[index]!;
                    TTargetItem targetItem = TargetList[targetIndex];

                    if (!AreEquivalentFunc(sourceItem, targetItem))
                        TargetList[targetIndex] = ProjectionFunc(sourceItem);

                    ItemSubscriptionManager.Unsubscribe((TSourceItem)oldItems[index]!);
                    ItemSubscriptionManager.Subscribe(sourceItem);
                }
            }
        }

        protected override void HandleMove(
            INotifyCollectionChanged? sender,
            IList movedItems,
            int oldStartIndex,
            int newStartIndex)
        {
            if (IsReentrant)
                return;

            using (BlockReentrancy())
            {
                if (movedItems.Count == 1 && TargetList is ObservableCollection<TTargetItem> observableTarget)
                {
                    observableTarget.Move(oldStartIndex, newStartIndex);
                }
                else
                {
                    // Multi-item moves are not supported by ObservableCollection, so we fall back to
                    // remove + insert. This means the target collection will raise Remove and Add
                    // notifications rather than a single Move notification.
                    List<TTargetItem> itemsToMove = new(movedItems.Count);

                    for (int index = 0; index < movedItems.Count; index++)
                    {
                        itemsToMove.Add(TargetList[oldStartIndex]);
                        TargetList.RemoveAt(oldStartIndex);
                    }

                    for (int index = 0; index < itemsToMove.Count; index++)
                        TargetList.Insert(newStartIndex + index, itemsToMove[index]);
                }
            }
        }

        protected override void HandleReset(INotifyCollectionChanged? sender)
        {
            if (IsReentrant)
            {
                ItemSubscriptionManager.UnsubscribeAll();

                foreach (TSourceItem item in SourceList)
                    ItemSubscriptionManager.Subscribe(item);

                return;
            }

            using (BlockReentrancy())
            {
                ItemSubscriptionManager.UnsubscribeAll();

                TargetList.Clear();

                foreach (TSourceItem item in SourceList)
                {
                    TTargetItem projectedItem = ProjectionFunc(item);
                    TargetList.Add(projectedItem);
                    ItemSubscriptionManager.Subscribe(item);
                }
            }
        }

        #endregion

        #region Private

        private IList<TSourceItem> SourceList { get; }

        private IList<TTargetItem> TargetList { get; }

        private INotifyCollectionChanged SourceChangeNotifier { get; }

        private Func<TSourceItem, TTargetItem> ProjectionFunc { get; }

        private Func<TSourceItem, TTargetItem, bool> AreEquivalentFunc { get; }

        private SubscriptionManager ItemSubscriptionManager { get; }

        private ScopedCounter ReentrancyGate { get; }

        private bool IsReentrant => ReentrancyGate.Count > 0;

        private IDisposable BlockReentrancy() => ReentrancyGate.EnterScope();

        private void HandleItemPropertyChanged(object? sender, PropertyChangedEventArgs args)
        {
            if (IsReentrant)
                return;

            using (BlockReentrancy())
            {
                TSourceItem sourceItem = (TSourceItem)sender!;

                int index = SourceList.IndexOf(sourceItem);

                TTargetItem targetItem = TargetList[index];

                if (!AreEquivalentFunc(sourceItem, targetItem))
                    TargetList[index] = ProjectionFunc(sourceItem);
            }
        }

        private void SubscribeNewItems(IList newItems)
        {
            for (int index = 0; index < newItems.Count; index++)
                ItemSubscriptionManager.Subscribe((TSourceItem)newItems[index]!);
        }

        private void UnsubscribeOldItems(IList oldItems)
        {
            for (int index = 0; index < oldItems.Count; index++)
                ItemSubscriptionManager.Unsubscribe((TSourceItem)oldItems[index]!);
        }

        private void ReplaceSubscriptions(IList oldItems, IList newItems)
        {
            for (int index = 0; index < newItems.Count; index++)
            {
                ItemSubscriptionManager.Unsubscribe((TSourceItem)oldItems[index]!);
                ItemSubscriptionManager.Subscribe((TSourceItem)newItems[index]!);
            }
        }

        private sealed class SubscriptionManager
        {
            public SubscriptionManager(PropertyChangedEventHandler eventHandler)
            {
                EventHandler = eventHandler;
            }

            public void Subscribe(TSourceItem? item)
            {
                if (item is not INotifyPropertyChanged notifier)
                    return;

                if (SubscribedItems.Contains(notifier))
                    return;

                notifier.PropertyChanged += EventHandler;
                SubscribedItems.Add(notifier);
            }

            public void Unsubscribe(TSourceItem? item)
            {
                if (item is not INotifyPropertyChanged notifier)
                    return;

                if (SubscribedItems.Remove(notifier))
                    notifier.PropertyChanged -= EventHandler;
            }

            public void UnsubscribeAll()
            {
                foreach (INotifyPropertyChanged notifier in SubscribedItems)
                    notifier.PropertyChanged -= EventHandler;

                SubscribedItems.Clear();
            }

            private PropertyChangedEventHandler EventHandler { get; }

            private List<INotifyPropertyChanged> SubscribedItems { get; } = [];
        }

        #endregion
    }

    #endregion
}