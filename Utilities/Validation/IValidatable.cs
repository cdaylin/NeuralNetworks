using System.ComponentModel;

namespace Daylin.Utilities.Validation;

/// <summary>
/// Indicates that an object is a validatable data entity, and provides access to an
/// <see cref="IObservableErrorNotifier"/> object that provides validation error information for the data entity.
/// </summary>
public interface IValidatable : INotifyPropertyChanged
{
    IObservableErrorNotifier ErrorNotifier { get; }
}

/// <summary>
/// An extension of <see cref="IValidatable"/> with specifically-typed errors.
/// </summary>
/// <typeparam name="TError">
/// Error type.
/// </typeparam>
public interface IValidatable<out TError> : IValidatable
{
    IObservableErrorNotifier IValidatable.ErrorNotifier => ErrorNotifier;

    /// <summary>
    /// Provides validation error information for this object.
    /// </summary>
    new IObservableErrorNotifier<TError> ErrorNotifier { get; }
}
