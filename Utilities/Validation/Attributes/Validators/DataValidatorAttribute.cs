namespace Daylin.Utilities.Validation.Attributes.Validators;

/// <summary>
/// An attribute that acts as a factory for constructing <see cref="IDataValidator{TError}">data validator
/// objects</see> for validating one or more properties of a data entity.
/// </summary>
/// <typeparam name="TError">
/// Type of error returned by the validator.
/// </typeparam>
public abstract class DataValidatorAttribute<TError> : Attribute
{
    /// <summary>
    /// Constructs a <see cref="DataValidatorAttribute{TError}"/> object.
    /// </summary>
    /// <param name="canShareValidator">
    /// Indicates whether a single validator object can be safely used to validate multiple data entities.
    /// (See <see cref="CanShareValidator"/>.)
    /// </param>
    protected DataValidatorAttribute(bool canShareValidator)
    {
        CanShareValidator = canShareValidator;
    }

    /// <summary>
    /// Indicates whether a single validator object is permitted to validate multiple data entities.
    /// </summary>
    /// <remarks>
    /// When <c>true</c>, a single instance of the validator is stored and reused for all instances of the
    /// validated type.  When <c>false</c>, a new validator instance is created for each instance of the
    /// validated type.
    /// </remarks>
    public bool CanShareValidator { get; }

    /// <summary>
    /// Returns a data validator object.
    /// </summary>
    /// <returns>
    /// Returns a data validator object.
    /// </returns>
    public virtual IDataValidator<TError> GetValidator()
    {
        if (CanShareValidator && SharedValidator is null)
            SharedValidator = CreateValidator();

        if (CanShareValidator)
            return SharedValidator!;

        return CreateValidator();
    }

    protected abstract IDataValidator<TError> CreateValidator();

    private IDataValidator<TError>? SharedValidator { get; set; }
}
