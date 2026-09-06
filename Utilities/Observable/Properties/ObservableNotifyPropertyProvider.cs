using System.ComponentModel;
using System.Reflection;

namespace Daylin.Utilities.Observable.Properties;

/// <summary>
/// Responsible for providing observable proxies for properties on objects that implement
/// <see cref="INotifyPropertyChanged"/>.
/// </summary>
public sealed class ObservableNotifyPropertyProvider : IObservablePropertyProvider
{
    public ObservableNotifyPropertyProvider()
    {
    }

    public bool IsPropertySupported(Type sourceType, string propertyName)
    {
        sourceType.ThrowIfNull();
        propertyName.ThrowIfNullOrEmptyOrWhitespace();

        if (!typeof(INotifyPropertyChanged).IsAssignableFrom(sourceType))
            return false;

        if (sourceType.GetProperty(propertyName, BindingFlags.Public | BindingFlags.Instance) is null)
            return false;

        return true;
    }

    IObservableProperty<T>? IObservablePropertyProvider.Create<T>(object source, string propertyName)
    {
        source.ThrowIfNull();

        if (source is not INotifyPropertyChanged notifyPropertyChanged)
            return null;

        return Create<T>(notifyPropertyChanged, propertyName);
    }

    public IObservableProperty<T>? Create<T>(INotifyPropertyChanged source, string propertyName)
    {
        source.ThrowIfNull();

        return new ObservableNotifyProperty<T>(source, propertyName);
    }
}
