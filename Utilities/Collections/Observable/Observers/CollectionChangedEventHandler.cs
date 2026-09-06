using System.Collections;
using System.Collections.Specialized;

namespace Daylin.Utilities.Collections.Observable.Observers;

/// <summary>
/// An abstract base class that assists in the handling of the
/// <see cref="INotifyCollectionChanged.CollectionChanged"/> event.
/// </summary>
/// <remarks>
/// This class helps to handle the <see cref="INotifyCollectionChanged.CollectionChanged"/> event by delegating
/// to a distinct handler for each type of <see cref="NotifyCollectionChangedAction"/>.  Each handler is a virtual
/// method that can be overridden to handle a specific type of <see cref="NotifyCollectionChangedAction"/>.
/// </remarks>
public abstract class CollectionChangedEventHandler
{
    public virtual void HandleCollectionChanged(object? sender, NotifyCollectionChangedEventArgs args)
    {
        INotifyCollectionChanged notifier = (sender as INotifyCollectionChanged)!;

        // The descriptions below of the actions and argument values have been taken from the following blog entry
        //  written by Stephen Cleary:  https://blog.stephencleary.com/2009/07/interpreting-notifycollectionchangedeve.html.
        switch (args.Action)
        {
            case NotifyCollectionChangedAction.Add:
                // NewItems contains the items that were added. If NewStartingIndex is not - 1, then it contains the
                //  index where the new items were added.
                HandleAdd(notifier, args.NewItems!, args.NewStartingIndex == -1 ? 0 : args.NewStartingIndex);
                break;

            case NotifyCollectionChangedAction.Remove:
                // OldItems contains the items that were removed. If OldStartingIndex is not - 1, then it contains the
                //  index where the old items were removed.
                HandleRemove(notifier, args.OldItems!, args.OldStartingIndex == -1 ? 0 : args.OldStartingIndex);
                break;

            case NotifyCollectionChangedAction.Replace:
                // OldItems contains the replaced items and NewItems contains the replacement items. NewStartingIndex and
                //  OldStartingIndex are equal.  If they are not -1, then they contain the index where the items were replaced.
                int startIndex = args.NewStartingIndex == -1 ? 0 : args.NewStartingIndex;
                HandleReplace(notifier, args.OldItems!, args.NewItems!, startIndex);
                break;

            case NotifyCollectionChangedAction.Move:
                // NewItems and OldItems are logically equivalent, and they contain the items that moved. OldStartingIndex
                //  contains the index where the items were moved from, and NewStartingIndex contains the index where the
                //  items were moved to.  A Move operation is logically treated as a Remove followed by an Add, so
                //  NewStartingIndex is interpreted as though the items had already been removed.
                HandleMove(notifier, args.NewItems!, args.OldStartingIndex, args.NewStartingIndex);
                break;

            case NotifyCollectionChangedAction.Reset:
                HandleReset(notifier);
                break;
        }
    }

    protected virtual void HandleAdd(INotifyCollectionChanged? sender, IList newItems, int startIndex)
    {
    }

    protected virtual void HandleRemove(INotifyCollectionChanged? sender, IList oldItems, int startIndex)
    {
    }

    protected virtual void HandleReplace(INotifyCollectionChanged? sender, IList oldItems, IList newItems, int startIndex)
    {
    }

    protected virtual void HandleMove(INotifyCollectionChanged? sender, IList movedItems, int oldStartIndex, int newStartIndex)
    {
    }

    protected virtual void HandleReset(INotifyCollectionChanged? sender)
    {
    }
}