using Daylin.Utilities.Collections.Observable.Observers;
using Daylin.Utilities.Observable;

using System.Collections.Specialized;
using System.ComponentModel;

namespace Daylin.Utilities.Validation.Observers;

public sealed class CollectionHasErrorsObserver : ObservableObject
{
    #region Construction

    public CollectionHasErrorsObserver(INotifyCollectionChanged collectionChangedNotifier)
        : this(collectionChangedNotifier, (IEnumerable<IValidatable>)collectionChangedNotifier)
    {
    }

    public CollectionHasErrorsObserver(
        INotifyCollectionChanged collectionChangedNotifier,
        IEnumerable<IValidatable> collection)
    {
        collectionChangedNotifier.ThrowIfNull();
        collection.ThrowIfNull();

        CollectionChangedNotifier = collectionChangedNotifier;
        Collection = collection;

        CollectionObserver = new CollectionObserver<IValidatable>(collectionChangedNotifier, collection);
        CollectionObserver.ItemAdded += HandleItemAdded;
        CollectionObserver.ItemRemoved += HandleItemRemoved;

        RegisterForErrorsChanged(Collection);
    }

    #endregion

    #region Public

    public event EventHandler? HasErrorsChanged;

    public bool HasErrors
    {
        get;

        set
        {
            if (SetProperty(ref field, value))
                HasErrorsChanged?.Invoke(this, EventArgs.Empty);
        }
    }

    public int ErrorCount
    {
        get;

        set
        {
            if (SetProperty(ref field, value))
                HasErrors = field > 0;
        }
    }

    #endregion

    #region Protected

    protected override void ReleaseResources()
    {
        CollectionObserver.Dispose();

        UnregisterForErrorsChanged(Collection);

        base.ReleaseResources();
    }

    #endregion

    #region Private

    private INotifyCollectionChanged CollectionChangedNotifier { get; }

    private IEnumerable<IValidatable> Collection { get; }

    private CollectionObserver<IValidatable> CollectionObserver { get; set; }

    private void RegisterForErrorsChanged(IEnumerable<IValidatable> items)
    {
        foreach (IValidatable item in items)
            RegisterForErrorsChanged(item);
    }

    private void UnregisterForErrorsChanged(IEnumerable<IValidatable> items)
    {
        foreach (IValidatable item in items)
            UnregisterForErrorsChanged(item);
    }

    private void RegisterForErrorsChanged(IValidatable item)
    {
        item.ErrorNotifier.ErrorsChanged += HandleErrorsChanged;
    }

    private void UnregisterForErrorsChanged(IValidatable item)
    {
        item.ErrorNotifier.ErrorsChanged -= HandleErrorsChanged;
    }

    private void HandleItemAdded(object? sender, CollectionChangedEventArgs<IValidatable> args)
    {
        RegisterForErrorsChanged(args.Item);

        if (args.Item.ErrorNotifier.HasErrors)
            ErrorCount += 1;
    }

    private void HandleItemRemoved(object? sender, CollectionChangedEventArgs<IValidatable> args)
    {
        UnregisterForErrorsChanged(args.Item);

        if (args.Item.ErrorNotifier.HasErrors)
            ErrorCount -= 1;
    }

    private void HandleErrorsChanged(object? sender, DataErrorsChangedEventArgs args)
    {
        IObservableErrorNotifier errorNotifier = (IObservableErrorNotifier)sender!;

        ErrorCount += errorNotifier.HasErrors ? 1 : -1;
    }

    #endregion
}
