using System.ComponentModel;

namespace Daylin.Utilities.Validation.Triggers.PropertyTriggers;

/// <summary>
/// A validation trigger that triggers validation in response to a property value changing.
/// </summary>
public sealed class PropertyChangedValidationTrigger : ValidationTrigger
{
    #region Construction

    /// <summary>
    /// Constructs a trigger that monitors all public properties for changes, triggering validation when any
    /// property value has changed.
    /// </summary>
    /// <param name="dataEntity">
    /// The data entity to monitor for property changes.
    /// </param>
    public PropertyChangedValidationTrigger(INotifyPropertyChanged dataEntity)
        : this(dataEntity, propertyName => propertyName.ToEnumerable())
    {
    }

    /// <summary>
    /// Constructs a trigger that monitors a single property for changes, triggering validation for a set
    /// of specified properties when the monitored property's value has changed.
    /// </summary>
    /// <param name="dataEntity">
    /// The data entity to monitor for property changes.
    /// </param>
    /// <param name="changedPropertyName">
    /// Name of the property to monitor for changes.
    /// </param>
    /// <param name="validatedPropertyNames">
    /// Names of properties to validate in response to the monitored property changing.
    /// </param>
    public PropertyChangedValidationTrigger(
        INotifyPropertyChanged dataEntity,
        string changedPropertyName,
        IEnumerable<string> validatedPropertyNames)
        : this(
              dataEntity,
              propertyName => propertyName == changedPropertyName ? validatedPropertyNames : Enumerable.Empty<string>())
    {
    }

    /// <summary>
    /// Constructs a trigger that monitors all public properties for changes, triggering validation for
    /// zero-to-many properties when a property value has changed.
    /// </summary>
    /// <param name="dataEntity">
    /// The data entity to monitor for property changes.
    /// </param>
    /// <param name="propertyNameMapper">
    /// A function that accepts the name of a property that has changed and returns a sequence of property
    /// names for which validation should be triggered.
    /// </param>
    public PropertyChangedValidationTrigger(
        INotifyPropertyChanged dataEntity,
        Func<string, IEnumerable<string>> propertyNameMapper)
    {
        dataEntity.ThrowIfNull();
        propertyNameMapper.ThrowIfNull();

        DataEntity = dataEntity;
        PropertyNameMapper = propertyNameMapper;

        DataEntity.PropertyChanged += HandlePropertyChanged;
    }

    #endregion

    #region Protected

    protected override void ReleaseResources()
    {
        DataEntity.PropertyChanged -= HandlePropertyChanged;

        DataEntity = null!;
        PropertyNameMapper = null!;

        base.ReleaseResources();
    }

    #endregion

    #region Private

    private INotifyPropertyChanged DataEntity { get; set; }

    private Func<string, IEnumerable<string>> PropertyNameMapper { get; set; }

    private void HandlePropertyChanged(object? sender, PropertyChangedEventArgs args)
    {
        if (args.PropertyName is null)
            return;

        foreach (string propertyName in PropertyNameMapper(args.PropertyName))
            RaiseValidationTriggered(propertyName);
    }

    #endregion
}
