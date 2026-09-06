using Daylin.Utilities.Observable.Properties;

using System.ComponentModel;
using System.Linq.Expressions;

namespace Daylin.Utilities.Extensions;

/// <summary>
/// Extension methods for type <see cref="INotifyPropertyChanged"/>.
/// </summary>
public static class NotifyPropertyChangedExtensions
{
    public static IObservableProperty<T> ObserveProperty<T>(
        this INotifyPropertyChanged source,
        string propertyName,
        ObservablePropertyFactory? factory = null)
    {
        source.ThrowIfNull();
        propertyName.ThrowIfNullOrEmptyOrWhitespace();

        factory ??= ObservablePropertyFactory.Global;

        return factory.Observe<T>(source, propertyName);
    }

    public static IObservableProperty<T> ObserveProperty<T>(
        this INotifyPropertyChanged source,
        PropertyPath path,
        T fallbackValue = default!,
        ObservablePropertyFactory? factory = null)
    {
        source.ThrowIfNull();
        path.ThrowIfNull();

        factory ??= ObservablePropertyFactory.Global;

        return factory.Observe(source, path, fallbackValue);
    }

    public static IObservableProperty<T> ObserveProperty<T>(
        this INotifyPropertyChanged source,
        Expression<Func<object, T>> propertyPathExpression,
        T fallbackValue = default!,
        ObservablePropertyFactory? factory = null)
    {
        source.ThrowIfNull();
        propertyPathExpression.ThrowIfNull();

        factory ??= ObservablePropertyFactory.Global;

        return factory.Observe<T>(source, propertyPathExpression, fallbackValue);
    }

    public static IObservableProperty<T> ObserveProperty<T, TSource>(
        this TSource source,
        Expression<Func<TSource, T>> propertyPathExpression,
        T fallbackValue = default!,
        ObservablePropertyFactory? factory = null)
        where TSource : INotifyPropertyChanged
    {
        source.ThrowIfNull();
        propertyPathExpression.ThrowIfNull();

        factory ??= ObservablePropertyFactory.Global;

        return factory.Observe<T, TSource>(source,  propertyPathExpression, fallbackValue);
    }
}
