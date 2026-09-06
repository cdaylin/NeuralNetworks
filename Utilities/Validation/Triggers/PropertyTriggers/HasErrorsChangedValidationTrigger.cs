using System.ComponentModel;

namespace Daylin.Utilities.Validation.Triggers.PropertyTriggers;

/// <summary>
/// A validation trigger that triggers validation when a property of type <see cref="IValidatable"/>
/// has a value that raises the <see cref="INotifyPropertyChanged.PropertyChanged"/> event for its
/// <see cref="INotifyDataErrorInfo.HasErrors"/> property.
/// </summary>
public class HasErrorsChangedValidationTrigger : PropertyValidationTrigger<IValidatable>
{
    #region Construction

    public HasErrorsChangedValidationTrigger(INotifyPropertyChanged dataEntity, string propertyName)
        : base(dataEntity, propertyName)
    {
        PropertyName = propertyName;

        RegisterForErrorNotifierPropertyChanged(PropertyValue);
    }

    #endregion

    #region Protected

    protected override void ReleaseResources()
    {
        UnregisterForErrorNotifierPropertyChanged(PropertyValue);

        base.ReleaseResources();
    }

    protected override void OnPropertyChanged(IValidatable oldValue, IValidatable newValue)
    {
        UnregisterForErrorNotifierPropertyChanged(oldValue);
        RegisterForErrorNotifierPropertyChanged(newValue);
    }

    protected string PropertyName { get; }

    #endregion

    #region Private

    private void RegisterForErrorNotifierPropertyChanged(IValidatable? validatablePropertyValue)
    {
        if (validatablePropertyValue is null)
            return;

        validatablePropertyValue.ErrorNotifier.PropertyChanged += HandleErrorNotifierPropertyChanged;
    }

    private void UnregisterForErrorNotifierPropertyChanged(IValidatable? validatablePropertyValue)
    {
        if (validatablePropertyValue is null)
            return;

        validatablePropertyValue.ErrorNotifier.PropertyChanged -= HandleErrorNotifierPropertyChanged;
    }

    private void HandleErrorNotifierPropertyChanged(object? sender, PropertyChangedEventArgs args)
    {
        if (args.PropertyName == nameof(INotifyDataErrorInfo.HasErrors))
            RaiseValidationTriggered(PropertyName);
    }

    #endregion
}
