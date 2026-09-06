using Daylin.Utilities.Observable;
using Daylin.Utilities.Validation.Manager;

namespace Daylin.Utilities.Validation;

/// <summary>
/// A data entity with validatable properties, exposing an <see cref="IObservableErrorNotifier"/> object that
/// provides validation error information.
/// </summary>
/// <typeparam name="TError">
/// Type of validation errors.
/// </typeparam>
public abstract class ValidatableObject<TError>
    : ObservableObject, IValidatable<TError>
{
    /// <summary>
    /// Constructs a validatable object.
    /// </summary>
    protected ValidatableObject()
    {
    }

    /// <summary>
    /// Provides validation error information for this object.
    /// </summary>
    /// <remarks>
    /// This property is lazily initialized, meaning that the property value is constructed and assigned the
    /// first time the property is accessed.  All validatable properties are validated during initialization.
    /// <para>
    /// It is strongly recommended that subclasses of <see cref="ValidatableObject{TError}"/> should not access
    /// this property during construction!  In general, property validation should be performed on an object
    /// only after the object has been fully constructed, in order to ensure that properties declared by
    /// subclasses have all been initialized.  The C# language currently provides no way to make a synchronous
    /// method call from within a constructor of a non-sealed class that can be guaranteed to execute only after
    /// constructors of all derived types have completed execution.  One way to ensure that the initial
    /// validation is performed after the validated object is fully constructed is to make all constructors
    /// non-public and to encapsulate both construction and validation into factory methods.  Lazy initialization
    /// is another way to ensure that validation can occur only after construction has completed -- with the
    /// one caveat that lazy initialization is not initiated during construction.
    /// </para>
    /// </remarks>
    public IObservableErrorNotifier<TError> ErrorNotifier
    {
        get
        {
            if (validationManager is null)
                SetProperty(ref validationManager, CreateValidationManager());

            return validationManager!;
        }
    }

    private ValidationManager<TError>? validationManager;

    protected override void ReleaseResources()
    {
        validationManager?.Dispose();

        base.ReleaseResources();
    }

    protected virtual ValidationManager<TError> CreateValidationManager()
    {
        return new ValidationManagerBuilder<TError>(this).Build();
    }
}
