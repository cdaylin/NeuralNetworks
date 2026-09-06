namespace Daylin.Utilities.Validation.Attributes.Validators;

/// <summary>
/// An attribute that acts as a factory for constructing <see cref="IDataValidator{TError}">data validator
/// objects</see> for validating instances of a data entity class.
/// </summary>
/// <typeparam name="TError">
/// Type of error returned by the validator.
/// </typeparam>
/// <typeparam name="TValidator">
/// Validator type.
/// </typeparam>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = true)]
public class EntityValidatorAttribute<TError, TValidator> : DataValidatorAttribute<TError>
    where TValidator : IDataValidator<TError>, new()
{
    /// <summary>
    /// Constructs a <see cref="EntityValidatorAttribute{TError, TValidator}"/> object.
    /// </summary>
    /// <param name="canShareValidator">
    /// Indicates whether a single validator object can be safely used to validate multiple data entities.
    /// (See <see cref="DataValidatorAttribute{TError}.CanShareValidator"/>.)
    /// </param>
    public EntityValidatorAttribute(bool canShareValidator = true)
        : base(canShareValidator)
    {
    }

    /// <summary>
    /// Returns a new <typeparamref name="TValidator"/> object.
    /// </summary>
    /// <returns>
    /// Returns a new <typeparamref name="TValidator"/> object.
    /// </returns>
    protected override IDataValidator<TError> CreateValidator() => new TValidator();
}
