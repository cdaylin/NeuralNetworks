using Daylin.Utilities.Events;
using Daylin.Utilities.Observable;
using Daylin.Utilities.Validation.Triggers;

using System.ComponentModel;

namespace Daylin.Utilities.Validation.Manager;

public class ValidationManager<TError>
    : ObservableObject, IObservableErrorNotifier<TError>
{
    #region Construction

    public ValidationManager(
        object dataEntity,
        IEnumerable<IDataValidator<TError>>? validators,
        IEnumerable<IValidationTrigger>? triggers)
    {
        dataEntity.ThrowIfNull();

        validators ??= Enumerable.Empty<IDataValidator<TError>>();
        triggers ??= Enumerable.Empty<IValidationTrigger>();

        DataEntity = dataEntity;

        foreach (IDataValidator<TError> validator in validators)
            AddValidator(validator);

        foreach (IValidationTrigger trigger in triggers)
            AddTrigger(trigger);

        DataErrorNotifier.PropertyChanged += HandleDataErrorNotifierPropertyChanged;
        DataErrorNotifier.ErrorsChanged += HandleDataErrorNotifierErrorsChanged;
    }

    #endregion

    #region Public

    /// <summary>
    /// Raised when validation errors for the entity or a property have changed.
    /// </summary>
    public event EventHandler<DataErrorsChangedEventArgs>? ErrorsChanged
    {
        add
        {
            ThrowExceptionIfDisposed();
            ErrorsChangedEventManager.Event += value;
        }

        remove => ErrorsChangedEventManager.Event -= value;
    }

    public IEnumerable<TError> GetErrors(string? propertyName)
    {
        ThrowExceptionIfDisposed();

        return DataErrorNotifier.GetErrors(propertyName);
    }

    public bool HasErrors
    {
        get
        {
            ThrowExceptionIfDisposed();

            return DataErrorNotifier.HasErrors;
        }
    }

    public void Validate()
    {
        ThrowExceptionIfDisposed();

        Validate(null);

        foreach (string propertyName in GetValidatedPropertyNames())
            Validate(propertyName);
    }

    public void Validate(string? propertyName)
    {
        ThrowExceptionIfDisposed();

        DataErrorNotifier.SetErrors(propertyName, RunValidators(propertyName));
    }

    #endregion

    #region Protected

    protected override void ReleaseResources()
    {
        foreach (IValidationTrigger trigger in Triggers)
            trigger.ValidationTriggered -= HandleValidationTriggered;

        Triggers.Clear();
        Validators.Clear();

        ErrorsChangedEventManager.Dispose();

        base.ReleaseResources();
    }

    protected virtual void RaiseErrorsChangedEvent(string? propertyName)
    {
        RaiseErrorsChangedEvent(new DataErrorsChangedEventArgs(propertyName));
    }

    protected virtual void RaiseErrorsChangedEvent(DataErrorsChangedEventArgs args)
    {
        ThrowExceptionIfDisposed();

        args.ThrowIfNull();

        ErrorsChangedEventManager.RaiseEvent(this, args);
    }

    protected object DataEntity { get; }

    protected List<IDataValidator<TError>> Validators { get; } = [];

    protected List<IValidationTrigger> Triggers { get; } = [];

    protected void AddValidator(IDataValidator<TError> validator)
    {
        ThrowExceptionIfDisposed();

        validator.ThrowIfNull();

        Validators.Add(validator);
    }

    protected void AddTrigger(IValidationTrigger trigger)
    {
        ThrowExceptionIfDisposed();

        trigger.ThrowIfNull();

        trigger.ValidationTriggered += HandleValidationTriggered;

        Triggers.Add(trigger);
    }

    /// <summary>
    /// Enumerates the names of all properties of <see cref="DataEntity"/> that can be validated.
    /// </summary>
    /// <remarks>
    /// The base implementation returns the names of all non-indexed public instance properties on the runtime
    /// (i.e.: concrete, actual) type of <see cref="DataEntity"/>.
    /// </remarks>
    /// <returns>
    /// Returns the names of all validated properties of <see cref="DataEntity"/>.
    /// </returns>
    protected virtual IEnumerable<string> GetValidatedPropertyNames()
    {
        return DataEntity
            .GetType()
            .GetPublicNonindexedInstanceProperties()
            .Select(property => property.Name);
    }

    protected virtual IEnumerable<TError> RunValidators(string? propertyName)
    {
        return Validators.SelectMany(validator => validator.GetErrors(DataEntity, propertyName));
    }

    protected virtual void HandleValidationTriggered(object? sender, ValidationTriggeredEventArgs args)
    {
        Validate(args.PropertyName);
    }

    #endregion

    #region Private

    private EventManager<DataErrorsChangedEventArgs> ErrorsChangedEventManager { get; } = new();

    private DataErrorNotifier<TError> DataErrorNotifier { get; } = new();

    private void HandleDataErrorNotifierPropertyChanged(object? sender, PropertyChangedEventArgs args)
    {
        if (args.PropertyName == nameof(HasErrors))
            RaisePropertyChangedEvent(nameof(HasErrors));
    }

    private void HandleDataErrorNotifierErrorsChanged(object? sender, DataErrorsChangedEventArgs args)
    {
        RaiseErrorsChangedEvent(args);
    }

    #endregion
}
