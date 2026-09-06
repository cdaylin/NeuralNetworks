using Daylin.Utilities.Validation.Triggers;

using System.ComponentModel;

namespace Daylin.Utilities.Validation.Attributes.Triggers;

/// <summary>
/// A property attribute that indicates that validation should be triggered by changes to the property value.
/// </summary>
[AttributeUsage(AttributeTargets.Property, AllowMultiple = true)]
public abstract class PropertyValidationTriggerAttribute : Attribute
{
    protected PropertyValidationTriggerAttribute()
    {
    }

    /// <summary>
    /// Returns a validation trigger object.
    /// </summary>
    /// <returns>
    /// Returns a validation trigger object.
    /// </returns>
    public virtual IValidationTrigger GetTrigger(INotifyPropertyChanged dataEntity)
    {
        dataEntity.ThrowIfNull();

        return CreateTrigger(dataEntity);
    }

    protected abstract IValidationTrigger CreateTrigger(INotifyPropertyChanged dataEntity);
}
