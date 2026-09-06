using Daylin.Utilities.Collections.Dictionaries;
using Daylin.Utilities.Events;
using Daylin.Utilities.Observable;

using System.ComponentModel;

namespace Daylin.Utilities.Validation;

/// <summary>
/// Responsible for tracking data validation errors for a data entity and for exposing those errors through
/// the interface <see cref="INotifyDataErrorInfo"/>.
/// </summary>
/// <remarks>
/// This class implements <see cref="INotifyPropertyChanged"/> in addition to <see cref="INotifyDataErrorInfo"/>,
/// for the purpose of raising <see cref="INotifyPropertyChanged.PropertyChanged"/> when the value of
/// <see cref="INotifyDataErrorInfo.HasErrors"/> changes.
/// </remarks>
/// <typeparam name="TError">
/// Type of errors tracked by the class.
/// </typeparam>
public class DataErrorNotifier<TError>
    : ObservableObject, IObservableErrorNotifier<TError>
{
    #region Construction

    public DataErrorNotifier()
    {
    }

    #endregion

    #region Public

    /// <summary>
    /// Raised when validation errors for the entity or a property have changed.
    /// </summary>
    public event EventHandler<DataErrorsChangedEventArgs>? ErrorsChanged
    {
        add => ErrorsChangedEventManager.Event += value;
        remove => ErrorsChangedEventManager.Event -= value;
    }

    /// <summary>
    /// Indicates whether any validation errors exist.
    /// </summary>
    public bool HasErrors
    {
        get;

        private set
        {
            if (SetProperty(ref field, value))
                OnHasErrorsChanged();
        }
    }

    /// <summary>
    /// Returns validation errors for the entity or a property on the entity.
    /// </summary>
    /// <param name="propertyName">
    /// Property name for which errors are returned, or <c>null</c> for the entity itself.
    /// </param>
    /// <returns>
    /// Returns validation error objects for the entity or for the specified property.
    /// </returns>
    public IEnumerable<TError> GetErrors(string? propertyName)
    {
        return Errors.GetValues(propertyName ?? EntityKey).ToArray();
    }

    /// <summary>
    /// Sets the error collection for the entity or a property on the entity.
    /// </summary>
    /// <param name="propertyName">
    /// Name of the property, or <c>null</c> for the entity itself.
    /// </param>
    /// <param name="errors">
    /// Errors for the property or entity.
    /// </param>
    /// <returns>
    /// Returns <c>true</c> if and only if the errors collection has changed for the property or entity.
    /// </returns>
    public bool SetErrors(string? propertyName, params IEnumerable<TError> errors)
    {
        errors.ThrowIfNull();

        IReadOnlyList<TError> errorsList = errors.ToList();

        string key = propertyName ?? EntityKey;

        if (errorsList.SequenceEqual(Errors.GetValues(key)))
            return false;

        Errors.RemoveKey(key);
        Errors.AddRange(key, errorsList);

        UpdateHasErrorsAndRaiseErrorsChangedEvent(propertyName);

        return true;
    }

    /// <summary>
    /// Adds errors for the entity or a property on the entity.
    /// </summary>
    /// <param name="propertyName">
    /// Name of the property, or <c>null</c>> for the entity itself.
    /// </param>
    /// <param name="errors">
    /// Errors to add.
    /// </param>
    /// <returns>
    /// Returns <c>true</c> if and only if the errors for the property or entity have changed.
    /// </returns>
    public bool AddErrors(string? propertyName, params IEnumerable<TError> errors)
    {
        errors.ThrowIfNull();

        if (!errors.Any())
            return false;

        Errors.AddRange(propertyName ?? EntityKey, errors);

        UpdateHasErrorsAndRaiseErrorsChangedEvent(propertyName);

        return true;
    }

    /// <summary>
    /// Removes specified errors for the entity or a property on the entity.
    /// </summary>
    /// <param name="propertyName">
    /// Name of the property, or <see cref="EntityKey"/> for the entity itself.
    /// </param>
    /// <param name="errors">
    /// Errors to remove.
    /// </param>
    /// <returns>
    /// Returns <c>true</c> if and only if the errors for the property or entity have changed.
    /// </returns>
    public bool RemoveErrors(string? propertyName, params IEnumerable<TError> errors)
    {
        errors.ThrowIfNull();

        bool isChanged = false;

        string key = propertyName ?? EntityKey;

        foreach (TError error in errors)
        {
            if (Errors.Remove(key, error))
                isChanged = true;
        }

        if (isChanged)
            UpdateHasErrorsAndRaiseErrorsChangedEvent(propertyName);

        return isChanged;
    }

    /// <summary>
    /// Removes all errors for the entity and all properties.
    /// </summary>
    /// <returns>
    /// Returns <c>true</c> if and only if at least one error was removed.
    /// </returns>
    public bool ClearErrors()
    {
        bool isChanged = false;

        // clear one key at a time, so that the errors-changed event can be raised for each property
        foreach (string propertyName in Errors.Keys)
        {
            if (ClearErrors(propertyName))
                isChanged = true;
        }

        return isChanged;
    }

    /// <summary>
    /// Removes all errors for the entity or a property on the entity.
    /// </summary>
    /// <param name="propertyName">
    /// Name of the property, or <c>null</c> for the entity itself.
    /// </param>
    /// <returns>
    /// Returns <c>true</c> if and only if at least one error was removed.
    /// </returns>
    public bool ClearErrors(string? propertyName)
    {
        bool isChanged = Errors.RemoveKey(propertyName ?? EntityKey);

        if (isChanged)
            UpdateHasErrorsAndRaiseErrorsChangedEvent(propertyName);

        return isChanged;
    }

    #endregion

    #region Protected

    protected override void ReleaseResources()
    {
        ErrorsChangedEventManager.Dispose();

        base.ReleaseResources();
    }

    /// <summary>
    /// Invokes <see cref="OnErrorsChanged(string)"/> and then raises the <see cref="ErrorsChanged"/> event.
    /// </summary>
    /// <param name="propertyName">
    /// Name of the property for which errors have changed, or <c>null</c> for the entity itself.
    /// </param>
    protected virtual void RaiseErrorsChangedEvent(string? propertyName)
    {
        ThrowExceptionIfDisposed();

        OnErrorsChanged(propertyName);

        ErrorsChangedEventManager.RaiseEvent(this, new DataErrorsChangedEventArgs(propertyName));
    }

    /// <summary>
    /// Invoked when the value of <see cref="HasErrors"/> has changed.
    /// </summary>
    protected virtual void OnHasErrorsChanged()
    {
    }

    /// <summary>
    /// Invoked when validation errors have changed for the entity or a property on the entity.
    /// </summary>
    /// <param name="propertyName">
    /// Name of the property for which errors have changed, or <c>null</c>> for the entity itself.
    /// </param>
    protected virtual void OnErrorsChanged(string? propertyName)
    {
    }

    #endregion

    #region Private

    private EventManager<DataErrorsChangedEventArgs> ErrorsChangedEventManager { get; } = new();

    private void UpdateHasErrorsAndRaiseErrorsChangedEvent(string? propertyName)
    {
        UpdateHasErrors();

        RaiseErrorsChangedEvent(propertyName ?? EntityKey);
    }

    private void UpdateHasErrors() => HasErrors = Errors.Keys.Any();

    /// <summary>
    /// Maps property names to associated errors.
    /// </summary>
    private ListMultiValueDictionary<string, TError> Errors { get; } = new();

    /// <summary>
    /// A key for mapping errors belonging to the entity itself.
    /// </summary>
    private string EntityKey { get; } = "this";

    #endregion
}
