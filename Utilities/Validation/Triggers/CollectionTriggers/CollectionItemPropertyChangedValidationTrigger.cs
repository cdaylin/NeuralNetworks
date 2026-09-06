using Daylin.Utilities.Collections.Observable.Observers;
using Daylin.Utilities.Observable.Properties;

using System.Collections.Specialized;
using System.ComponentModel;

namespace Daylin.Utilities.Validation.Triggers.CollectionTriggers;

/// <summary>
/// A validation trigger that monitors the contents of a collection, triggering validation when a property
/// has changed for an entity within the collection.
/// </summary>
/// <remarks>
/// The monitored collection must implement <see cref="INotifyCollectionChanged"/> and
/// <see cref="IEnumerable{TItem}"/>, and items in the collection must be distinct.
/// </remarks>
/// <typeparam name="TItem">
/// Type of items in the collection.
/// </typeparam>
public sealed class CollectionItemPropertyChangedValidationTrigger<TItem> : ValidationTrigger
    where TItem : notnull, INotifyPropertyChanged
{
    #region Construction

    /// <summary>
    /// Constructs a new validation trigger.
    /// </summary>
    /// <param name="propertyChangedNotifier">
    /// An observable data entity.
    /// </param>
    /// <param name="collectionPropertyName">
    /// Name of a property of a type that implements both <see cref="INotifyCollectionChanged"/> and
    /// <see cref="ICollection{TItem}"/>.
    /// </param>
    /// <param name="itemPropertyName">
    /// Names of a property on the items of type <typeparamref name="TItem"/> within the collection that should
    /// trigger validation when changed.
    /// </param>
    /// <param name="validatedPropertyName">
    /// Name of the property that should be validated when the property value changes for an item in the collection.
    /// When <c>null</c>, property <paramref name="collectionPropertyName"/> is validated.
    /// </param>
    public CollectionItemPropertyChangedValidationTrigger(
        INotifyPropertyChanged propertyChangedNotifier,
        string collectionPropertyName,
        string itemPropertyName,
        string? validatedPropertyName = null)
    {
        propertyChangedNotifier.ThrowIfNull();
        collectionPropertyName.ThrowIfNullOrEmpty();
        itemPropertyName.ThrowIfNullOrEmpty();
        validatedPropertyName.ThrowIfEmpty();

        PropertyChangedNotifier = propertyChangedNotifier;
        CollectionPropertyName = collectionPropertyName;
        ValidatedPropertyName = validatedPropertyName ?? collectionPropertyName;
        ItemPropertyName = itemPropertyName;

        PropertyObserver = new ObservableNotifyProperty<INotifyCollectionChanged?>(
            PropertyChangedNotifier,
            collectionPropertyName);

        PropertyObserver.ValueChanged += HandleValueChanged;

        RegisterForEvents(PropertyObserver.Value);
    }

    #endregion

    #region Protected

    protected override void ReleaseResources()
    {
        UnregisterForEvents(PropertyObserver.Value);

        PropertyObserver.Dispose();
        CollectionObserver?.Dispose();

        base.ReleaseResources();
    }

    #endregion

    #region Private

    private INotifyPropertyChanged PropertyChangedNotifier { get; }

    private string CollectionPropertyName { get; }

    private string ValidatedPropertyName { get; }

    private string ItemPropertyName { get; }

    private ObservableProperty<INotifyCollectionChanged?> PropertyObserver { get; }

    private CollectionObserver<TItem>? CollectionObserver { get; set; }

    private void HandleValueChanged(object? sender, PropertyChangedEventArgs<INotifyCollectionChanged?> args)
    {
        UnregisterForEvents(args.OldValue);
        RegisterForEvents(args.NewValue);
    }

    private void RegisterForEvents(INotifyCollectionChanged? propertyValue)
    {
        RegisterForCollectionChanged(PropertyObserver.Value);
        RegisterForPropertyChanged(PropertyObserver.Value as IEnumerable<TItem>);
    }

    private void UnregisterForEvents(INotifyCollectionChanged? propertyValue)
    {
        UnregisterForCollectionChanged(PropertyObserver.Value);
        UnregisterForPropertyChanged(PropertyObserver.Value as IEnumerable<TItem>);
    }

    private void RegisterForCollectionChanged(INotifyCollectionChanged? collection)
    {
        if (collection is null)
            return;

        CollectionObserver = new CollectionObserver<TItem>(collection);

        CollectionObserver.ItemAdded += HandleItemAdded;
        CollectionObserver.ItemRemoved += HandleItemRemoved;
    }

    private void UnregisterForCollectionChanged(INotifyCollectionChanged? collection)
    {
        if (collection is null)
            return;

        CollectionObserver!.Dispose();
        CollectionObserver = null;
    }

    private void RegisterForPropertyChanged(IEnumerable<TItem>? items)
    {
        if (items is null)
            return;

        foreach (TItem item in items)
            RegisterForPropertyChanged(item);
    }

    private void UnregisterForPropertyChanged(IEnumerable<TItem>? items)
    {
        if (items is null)
            return;

        foreach (TItem item in items)
            UnregisterForPropertyChanged(item);
    }

    private void RegisterForPropertyChanged(TItem item)
    {
        item.PropertyChanged += HandleItemPropertyChanged;
    }

    private void UnregisterForPropertyChanged(TItem item)
    {
        item.PropertyChanged -= HandleItemPropertyChanged;
    }

    private void HandleItemAdded(object? sender, CollectionChangedEventArgs<TItem> args)
    {
        RegisterForPropertyChanged(args.Item);
    }

    private void HandleItemRemoved(object? sender, CollectionChangedEventArgs<TItem> args)
    {
        UnregisterForPropertyChanged(args.Item);
    }

    private void HandleItemPropertyChanged(object? sender, PropertyChangedEventArgs args)
    {
        if (ItemPropertyName == args.PropertyName!)
            RaiseValidationTriggered(ValidatedPropertyName);
    }

    #endregion
}