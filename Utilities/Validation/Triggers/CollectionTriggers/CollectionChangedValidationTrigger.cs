using Daylin.Utilities.Validation.Triggers.PropertyTriggers;

using System.Collections.Specialized;
using System.ComponentModel;

namespace Daylin.Utilities.Validation.Triggers.CollectionTriggers;

/// <summary>
/// A validation trigger that monitors a property of type <see cref="INotifyCollectionChanged"/> for changes,
/// triggering validation when the contents of the collection have changed.
/// </summary>
public sealed class CollectionChangedValidationTrigger : PropertyValidationTrigger<INotifyCollectionChanged?>
{
    /// <summary>
    /// Constructs a new validation trigger.
    /// </summary>
    /// <param name="dataEntity">
    /// An observable data entity.
    /// </param>
    /// <param name="collectionPropertyName">
    /// Name of a property of type <see cref="INotifyCollectionChanged"/>.
    /// </param>
    /// <param name="validatedPropertyName">
    /// Name of the property that should be validated when the collection changes. When <c>null</c>,
    /// property <paramref name="collectionPropertyName"/> is validated.
    /// </param>
    /// <param name="shouldTriggerWhenReordered">
    /// Specifies whether validation should be triggered when items have been moved within the collection.
    /// </param>
    public CollectionChangedValidationTrigger(
        INotifyPropertyChanged dataEntity,
        string collectionPropertyName,
        string? validatedPropertyName = null,
        bool shouldTriggerWhenReordered = false)
        : base(dataEntity, collectionPropertyName)
    {
        collectionPropertyName.ThrowIfNullOrEmpty();
        validatedPropertyName.ThrowIfEmpty();

        CollectionPropertyName = collectionPropertyName;
        ValidatedPropertyName = validatedPropertyName ?? collectionPropertyName;
        ShouldTriggerWhenReordered = shouldTriggerWhenReordered;

        RegisterForCollectionChanged(PropertyValue);
    }

    protected override void ReleaseResources()
    {
        UnregisterForCollectionChanged(PropertyValue);

        base.ReleaseResources();
    }

    protected override void OnPropertyChanged(
        INotifyCollectionChanged? oldValue,
        INotifyCollectionChanged? newValue)
    {
        UnregisterForCollectionChanged(oldValue);
        RegisterForCollectionChanged(newValue);

        RaiseValidationTriggered(ValidatedPropertyName);
    }

    private string CollectionPropertyName { get; }

    private string ValidatedPropertyName { get; }

    private bool ShouldTriggerWhenReordered { get; }

    private void RegisterForCollectionChanged(INotifyCollectionChanged? collection)
    {
        if (collection is null)
            return;

        collection.CollectionChanged += HandleCollectionChanged;
    }

    private void UnregisterForCollectionChanged(INotifyCollectionChanged? collection)
    {
        if (collection is null)
            return;

        collection.CollectionChanged -= HandleCollectionChanged;
    }

    private void HandleCollectionChanged(object? sender, NotifyCollectionChangedEventArgs args)
    {
        if (args.Action is not NotifyCollectionChangedAction.Move || ShouldTriggerWhenReordered)
            RaiseValidationTriggered(ValidatedPropertyName);
    }
}
