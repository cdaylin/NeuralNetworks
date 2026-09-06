using Daylin.Utilities.Observable.Properties;

using System.ComponentModel;

namespace Daylin.Utilities.Validation.Triggers.PropertyTriggers;

/// <summary>
/// A validation trigger that observes a single property on a data entity for changes.
/// </summary>
/// <typeparam name="T">
/// The type of the property that is observed for changes.
/// </typeparam>
public abstract class PropertyValidationTrigger<T> : ValidationTrigger
{
    #region Construction

    /// <summary>
    /// Constructs a validation trigger.
    /// </summary>
    /// <param name="dataEntity">
    /// An observable data entity.
    /// </param>
    /// <param name="changedPropertyName">
    /// Name of the property to observe for changes.
    /// </param>
    protected PropertyValidationTrigger(INotifyPropertyChanged dataEntity, string changedPropertyName)
    {
        dataEntity.ThrowIfNull();
        changedPropertyName.ThrowIfNullOrEmptyOrWhitespace();

        TriggerProperty = new ObservableNotifyProperty<T>(dataEntity, changedPropertyName);
        TriggerProperty.ValueChanged += HandleValueChanged;

        PropertyValue = TriggerProperty.Value;
    }

    #endregion

    #region Protected

    protected override void ReleaseResources()
    {
        TriggerProperty.Dispose();

        base.ReleaseResources();
    }

    protected ObservableProperty<T> TriggerProperty { get; }

    /// <summary>
    /// Current value of the property being observed for changes.
    /// </summary>
    protected T PropertyValue
    {
        get
        {
            ThrowExceptionIfDisposed();
            return field;
        }

        private set;
    }

    /// <summary>
    /// Invoked when the observed property value has changed.
    /// </summary>
    /// <param name="oldValue">
    /// Previous property value.
    /// </param>
    /// <param name="newValue">
    /// Current property value.
    /// </param>
    protected abstract void OnPropertyChanged(T oldValue, T newValue);

    #endregion

    #region Private

    private void HandleValueChanged(object? sender, PropertyChangedEventArgs<T> args)
    {
        PropertyValue = args.NewValue;

        OnPropertyChanged(args.OldValue, args.NewValue);
    }

    #endregion
}
