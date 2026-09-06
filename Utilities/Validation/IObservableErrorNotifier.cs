using System.Collections;
using System.ComponentModel;

namespace Daylin.Utilities.Validation;

/// <summary>
/// Combines interfaces <see cref="INotifyDataErrorInfo"/> and <see cref="INotifyPropertyChanged"/> to provide
/// access to data errors.
/// </summary>
/// <remarks>
/// Interface <see cref="INotifyDataErrorInfo"/> should be derived from <see cref="INotifyPropertyChanged"/>,
/// but it is not.  If a type implements <see cref="INotifyDataErrorInfo"/> but not
/// <see cref="INotifyPropertyChanged"/>, then an observer cannot detect when the value of 
/// <see cref="INotifyDataErrorInfo.HasErrors"/> changes.  This interface extends both of those interfaces in
/// order to make that dependency explicit.
/// </remarks>
public interface IObservableErrorNotifier : INotifyPropertyChanged, INotifyDataErrorInfo
{
}

/// <summary>
/// An exension of <see cref="IObservableErrorNotifier"/> with strongly-typed errors.
/// </summary>
/// <typeparam name="TError">
/// Error type.
/// </typeparam>
public interface IObservableErrorNotifier<out TError> : IObservableErrorNotifier
{
    /// <summary>
    /// Returns validation errors for a property.
    /// </summary>
    /// <param name="propertyName">
    /// Property name.
    /// </param>
    /// <returns>
    /// Returns validation error objects for property "<paramref name="propertyName"/>".
    /// </returns>
    IEnumerable INotifyDataErrorInfo.GetErrors(string? propertyName) => GetErrors(propertyName);

    /// <summary>
    /// Returns validation errors for a property.
    /// </summary>
    /// <param name="propertyName">
    /// Property name.  When null, errors are returned for the entity itself (errors that apply to the entity but not
    /// to a specific property on that entity).
    /// </param>
    /// <returns>
    /// Returns validation error objects for the entity or for the specified property.
    /// </returns>
    new IEnumerable<TError> GetErrors(string? propertyName);
}