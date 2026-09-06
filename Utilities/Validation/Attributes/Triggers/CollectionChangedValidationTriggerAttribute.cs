using Daylin.Utilities.Validation.Triggers;
using Daylin.Utilities.Validation.Triggers.CollectionTriggers;

using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace Daylin.Utilities.Validation.Attributes.Triggers;

public class CollectionChangedValidationTriggerAttribute : PropertyValidationTriggerAttribute
{
    public CollectionChangedValidationTriggerAttribute(
        [CallerMemberName] string collectionPropertyName = "",
        string? validatedPropertyName = null,
        bool shouldTriggerWhenReordered = true)
    {
        collectionPropertyName.ThrowIfNullOrEmpty();
        validatedPropertyName.ThrowIfEmpty();

        CollectionPropertyName = collectionPropertyName;
        ValidatedPropertyName = validatedPropertyName ?? collectionPropertyName;
        ShouldTriggerWhenReordered = shouldTriggerWhenReordered;
    }

    protected string CollectionPropertyName { get; }

    protected string ValidatedPropertyName { get; }

    protected bool ShouldTriggerWhenReordered { get; }

    protected override IValidationTrigger CreateTrigger(INotifyPropertyChanged dataEntity)
    {
        return new CollectionChangedValidationTrigger(
            dataEntity,
            CollectionPropertyName,
            ValidatedPropertyName,
            ShouldTriggerWhenReordered);
    }
}
