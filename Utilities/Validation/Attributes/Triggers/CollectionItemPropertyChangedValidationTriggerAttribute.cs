using Daylin.Utilities.Validation.Triggers;
using Daylin.Utilities.Validation.Triggers.CollectionTriggers;

using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace Daylin.Utilities.Validation.Attributes.Triggers;

public class CollectionItemPropertyChangedValidationTriggerAttribute<TItem> : PropertyValidationTriggerAttribute
    where TItem : notnull, INotifyPropertyChanged
{
    public CollectionItemPropertyChangedValidationTriggerAttribute(
        string itemPropertyName,
        [CallerMemberName] string collectionPropertyName = "",
        string? validatedPropertyName = null)
    {
        itemPropertyName.ThrowIfNullOrEmpty();
        collectionPropertyName.ThrowIfNullOrEmpty();
        validatedPropertyName.ThrowIfEmpty();

        ItemPropertyName = itemPropertyName;
        CollectionPropertyName = collectionPropertyName;
        ValidatedPropertyName = validatedPropertyName ?? collectionPropertyName;
    }

    protected string ItemPropertyName { get; }

    protected string CollectionPropertyName { get; }

    protected string ValidatedPropertyName { get; }

    protected override IValidationTrigger CreateTrigger(INotifyPropertyChanged dataEntity)
    {
        return new CollectionItemPropertyChangedValidationTrigger<TItem>(
            dataEntity,
            CollectionPropertyName,
            ItemPropertyName,
            ValidatedPropertyName);
    }
}
