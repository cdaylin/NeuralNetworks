namespace Daylin.Utilities.Validation.Triggers;

/// <summary>
/// A validation trigger raises an event to indicate that validation should be performed for a data entity
/// or one of its properties.
/// </summary>
public interface IValidationTrigger
{
    event EventHandler<ValidationTriggeredEventArgs> ValidationTriggered;
}
