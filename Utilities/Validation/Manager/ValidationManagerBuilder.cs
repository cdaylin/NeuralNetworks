using Daylin.Utilities.Validation.Attributes.Triggers;
using Daylin.Utilities.Validation.Attributes.Validators;
using Daylin.Utilities.Validation.Triggers;
using Daylin.Utilities.Validation.Triggers.PropertyTriggers;

using System.ComponentModel;
using System.Reflection;

namespace Daylin.Utilities.Validation.Manager;

/// <summary>
/// Responsible for building <see cref="ValidationManager{TError}"/> objects.
/// </summary>
public class ValidationManagerBuilder<TError>
{
    #region Construction

    public ValidationManagerBuilder(INotifyPropertyChanged dataEntity)
    {
        dataEntity.ThrowIfNull();

        DataEntity = dataEntity;

        AddValidatorsViaReflection();
        AddTriggersViaReflection();
    }

    #endregion

    #region Public

    public INotifyPropertyChanged DataEntity { get; }

    public void AddValidator(IDataValidator<TError> validator)
    {
        validator.ThrowIfNull();

        Validators.Add(validator);
    }

    public void AddTrigger(IValidationTrigger trigger)
    {
        trigger.ThrowIfNull();

        Triggers.Add(trigger);
    }

    public ValidationManager<TError> Build()
    {
        ValidationManager<TError> validationManager = new(
            DataEntity,
            Validators,
            Triggers);

        validationManager.Validate();

        return validationManager;
    }

    #endregion

    #region Protected

    protected List<IDataValidator<TError>> Validators { get; } = [];

    protected List<IValidationTrigger> Triggers { get; } = [];

    protected virtual IEnumerable<IDataValidator<TError>> GetValidatorsViaReflection()
    {
        if (GetValidatorFromClassAttribute() is IDataValidator<TError> validator)
            yield return validator;
    }

    protected virtual IEnumerable<IValidationTrigger> GetTriggersViaReflection()
    {
        yield return new PropertyChangedValidationTrigger(DataEntity);

        foreach (IValidationTrigger trigger in GetTriggersFromPropertyAttributes())
            yield return trigger;
    }

    protected IDataValidator<TError>? GetValidatorFromClassAttribute()
    {
        DataValidatorAttribute<TError>? attribute = DataEntity
            .GetType()
            .GetCustomAttributes(inherit: true)
            .OfType<DataValidatorAttribute<TError>>()
            .FirstOrDefault();

        if (attribute is null)
            return null;

        return attribute.GetValidator();
    }

    protected IEnumerable<IValidationTrigger> GetTriggersFromPropertyAttributes()
    {
        foreach (PropertyInfo property in GetNonindexedPublicInstanceProperties())
        {
            if (property.PropertyType.IsAssignableTo(typeof(IValidatable)))
                yield return new HasErrorsChangedValidationTrigger(DataEntity, property.Name);

            foreach (IValidationTrigger trigger in GetValidationTriggers(property))
                yield return trigger;
        }
    }

    protected IEnumerable<IValidationTrigger> GetValidationTriggers(PropertyInfo property)
    {
        foreach (PropertyValidationTriggerAttribute attribute in property.GetCustomAttributes<PropertyValidationTriggerAttribute>())
            yield return attribute.GetTrigger(DataEntity);
    }

    protected IEnumerable<PropertyInfo> GetNonindexedPublicInstanceProperties()
    {
        return DataEntity.GetType().GetPublicNonindexedInstanceProperties();
    }

    #endregion

    #region Private

    private void AddValidatorsViaReflection()
    {
        foreach (IDataValidator<TError> validator in GetValidatorsViaReflection())
            AddValidator(validator);
    }

    private void AddTriggersViaReflection()
    {
        foreach (IValidationTrigger trigger in GetTriggersViaReflection())
            AddTrigger(trigger);
    }

    #endregion
}
