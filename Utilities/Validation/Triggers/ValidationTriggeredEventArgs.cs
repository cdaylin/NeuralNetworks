namespace Daylin.Utilities.Validation.Triggers;

public class ValidationTriggeredEventArgs : EventArgs
{
    public ValidationTriggeredEventArgs(string? propertyName)
    {
        PropertyName = propertyName;
    }

    public string? PropertyName { get; }
}
