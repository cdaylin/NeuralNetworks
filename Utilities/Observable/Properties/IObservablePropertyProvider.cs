namespace Daylin.Utilities.Observable.Properties;

/// <summary>
/// Responsible for providing observable proxies for properties.
/// </summary>
public interface IObservablePropertyProvider
{
    bool IsPropertySupported(Type sourceType, string propertyName);

    /// <summary>
    /// Creates an observable proxy for a property.
    /// </summary>
    /// <typeparam name="T">
    /// Type of the property.
    /// </typeparam>
    /// <param name="source">
    /// Object that contains the property.
    /// </param>
    /// <param name="propertyName">
    /// Name of the property.
    /// </param>
    /// <returns>
    /// Returns a new instance of <see cref="IObservableProperty{T}"/> that acts as a proxy for the property.
    /// Returns <c>null</c> when this factory cannot create a proxy for the property.
    /// </returns>
    IObservableProperty<T>? Create<T>(object source, string propertyName);
}