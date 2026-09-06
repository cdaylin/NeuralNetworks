using System.ComponentModel;

namespace Daylin.Utilities.Observable.Properties;

/// <summary>
/// An observable proxy to a property on an object of type <see cref="INotifyPropertyChanged"/>.
/// </summary>
/// <typeparam name="T">
/// Type of the property.
/// </typeparam>
public sealed class ObservableNotifyProperty<T> : ObservableProperty<T>, IObservableProperty<T>
{
    #region Construction

    public ObservableNotifyProperty(INotifyPropertyChanged source, string propertyName)
    {
        source.ThrowIfNull();
        propertyName.ThrowIfNullOrEmptyOrWhitespace();

        Source = source;
        PropertyName = propertyName;

        Value = GetPropertyValue();

        Source.PropertyChanged += HandlePropertyChanged;
    }

    #endregion

    #region Public

    public string PropertyName { get; }

    #endregion

    #region Protected

    protected override void ReleaseResources()
    {
        Source.PropertyChanged -= HandlePropertyChanged;
        Source = null!;

        base.ReleaseResources();
    }

    #endregion

    #region Private

    private INotifyPropertyChanged Source { get; set; }

    private T GetPropertyValue()
    {
        try
        {
            return Source.GetPropertyValue<T>(PropertyName);
        }
        catch (Exception exception)
        {
            throw new InvalidOperationException(
                $"Failed to get the value for property '{PropertyName}' from object of type '{Source.GetType()}'.",
                exception);
        }
    }

    private void HandlePropertyChanged(object? sender, PropertyChangedEventArgs args)
    {
        if (string.IsNullOrEmpty(args.PropertyName) || Equals(args.PropertyName, PropertyName))
            Value = GetPropertyValue();
    }

    #endregion
}