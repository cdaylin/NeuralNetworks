namespace Daylin.Utilities.Validation.Editing;

/// <summary>
/// A data entity with validatable properties and support for edit buffering.
/// </summary>
/// <typeparam name="TError">
/// Type of validation errors.
/// </typeparam>
/// <typeparam name="TSelf">
/// Current type (self-referential).
/// </typeparam>
public abstract class BufferableValidatableObject<TError, TSelf> : ValidatableObject<TError>, IEditBufferCloneSource<TSelf>
    where TSelf : BufferableValidatableObject<TError, TSelf>
{
    protected BufferableValidatableObject()
    {
    }

    protected BufferableValidatableObject(BufferableValidatableObject<TError, TSelf> source)
    {
        // This copy constructor is not strictly necessary, because this base class does not need to copy
        //  any data.  It is provided in order to allow for a consistent pattern: each derived class should
        //  implement a copy constructor that calls the copy constructor of the base class.
    }

    TSelf IEditBufferCloneSource<TSelf>.CreateEditBufferClone() => CreateEditBufferClone();

    /// <summary>
    /// Constructs and returns a new object that is a clone of this object for use in an edit buffer.
    /// </summary>
    /// <remarks>
    /// Each concrete derived class should override this method to call its own copy constructor.  The copy
    /// constructor should be implemented to call the copy constructor of its base class and then copy any data
    /// managed by the derived class.  After the clone has been created, additional initialization to prepare the
    /// clone for use in an edit buffer may be performed before returning the clone.
    /// </remarks>
    /// <returns>
    /// Returns a new clone of this object.
    /// </returns>
    /// <seealso cref="IEditBufferCloneSource{T}"/>
    protected abstract TSelf CreateEditBufferClone();
}
