using Daylin.Utilities.Validation.Triggers;
using Daylin.Utilities.Validation.Triggers.PropertyTriggers;

using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace Daylin.Utilities.Validation.Attributes.Triggers;

public class PropertyChangedValidationTriggerAttribute : PropertyValidationTriggerAttribute
{
    public PropertyChangedValidationTriggerAttribute(
        string validatedPropertyName,
        [CallerMemberName] string changedPropertyName = "")
        : this([validatedPropertyName], changedPropertyName)
    {
    }

    public PropertyChangedValidationTriggerAttribute(
        string[] validatedPropertyNames,
        [CallerMemberName] string changedPropertyName = "")
    {
        validatedPropertyNames.ThrowIfNull();
        validatedPropertyNames.ThrowIfEmpty();
        validatedPropertyNames.ThrowIfContainsNull();

        changedPropertyName.ThrowIfNullOrEmpty();

        ValidatedPropertyNames = (string[])validatedPropertyNames.Clone();
        ChangedPropertyName = changedPropertyName;
    }

    protected string ChangedPropertyName { get; }

    protected IEnumerable<string> ValidatedPropertyNames { get; }

    protected override IValidationTrigger CreateTrigger(INotifyPropertyChanged dataEntity)
    {
        return new PropertyChangedValidationTrigger(
            dataEntity,
            ChangedPropertyName,
            ValidatedPropertyNames);
    }
}
