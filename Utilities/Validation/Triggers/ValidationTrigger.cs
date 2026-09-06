using Daylin.Utilities.Disposable;

namespace Daylin.Utilities.Validation.Triggers;

/// <summary>
/// Implementation of <see cref="IValidationTrigger"/>.
/// </summary>
public abstract class ValidationTrigger : DisposableObject, IValidationTrigger
{
    protected ValidationTrigger()
    {
    }

    public event EventHandler<ValidationTriggeredEventArgs>? ValidationTriggered;

    /// <summary>
    /// Raises the <see cref="ValidationTriggered"/> event.
    /// </summary>
    /// <param name="propertyName">
    /// Name of the property for which validation should be triggered, or <c>null</c> for the data entity itself
    /// </param>
    protected void RaiseValidationTriggered(string? propertyName)
    {
        ValidationTriggered?.Invoke(
            this,
            new ValidationTriggeredEventArgs(propertyName));
    }
}
