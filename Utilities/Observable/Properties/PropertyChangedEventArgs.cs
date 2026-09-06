namespace Daylin.Utilities.Observable.Properties;

public class PropertyChangedEventArgs<T> : EventArgs
{
    public PropertyChangedEventArgs(T oldValue, T newValue)
    {
        OldValue = oldValue;
        NewValue = newValue;
    }

    public T OldValue { get; }

    public T NewValue { get; }
}
