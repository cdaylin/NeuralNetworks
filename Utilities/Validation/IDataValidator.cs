using System.ComponentModel;

namespace Daylin.Utilities.Validation;

/// <summary>
/// Responsible for calculating data validation errors for properties of a data entity.
/// </summary>
/// <remarks>
/// Validation errors can be associated with an individual property on the entity or with the entity itself.
/// The reason for allowing validation errors to be associated with the entity itself is to support interoperability
/// with <see cref="INotifyDataErrorInfo"/>. <see cref="INotifyDataErrorInfo.GetErrors(string?)"/> allows for errors
/// to be obtained for the entity as a whole when the property name is <c>null</c>.
/// </remarks>
public interface IDataValidator
{
    /// <summary>
    /// Gets validation errors for a data entity, optionally for a specific property.
    /// </summary>
    /// <param name="dataEntity">
    /// The data entity to validate.
    /// </param>
    /// <param name="propertyName">
    /// The name of the property to validate, or <c>null</c> to validate the entity as a whole.
    /// </param>
    /// <returns>
    /// Returns a sequence of validation errors.
    /// </returns>
    IEnumerable<object> GetErrors(object dataEntity, string? propertyName);
}

/// <inheritdoc cref="IDataValidator"/>/>
/// <typeparam name="TError">
/// Type of errors returned by the validator.
/// </typeparam>
public interface IDataValidator<out TError> : IDataValidator
{
    /// <inheritdoc cref="IDataValidator.GetErrors(object, string?)"/>
    new IEnumerable<TError> GetErrors(object dataEntity, string? propertyName);

    IEnumerable<object> IDataValidator.GetErrors(object dataEntity, string? propertyName)
    {
        return GetErrors(dataEntity, propertyName).Cast<object>();
    }
}
