namespace Daylin.Utilities.Validation.Editing;

/// <summary>
/// Indicates that an object can be cloned for use in an edit buffer (see <see cref="EditBuffer"/>).
/// </summary>
public interface IEditBufferCloneSource : IValidatable
{
    /// <summary>
    /// Creates a clone of the current instance that can be used in an edit buffer.
    /// </summary>
    /// <remarks>
    /// The clone should be of the same runtime type as the source object. The initial value of each publicly
    /// settable instance property on the clone should equal the value of the corresponding property on the source
    /// object. However, a change to the value of a publicly settable instance property on the clone should not be
    /// automatically applied to the source, nor vice versa. The edit buffer will manage copying values for public
    /// settable properties.
    /// </remarks>
    /// <returns>
    /// Returns a new object that is a clone of the current object for the purpose of edit buffering.
    /// </returns>
    IEditBufferCloneSource CreateEditBufferClone();
}

/// <summary>
/// Indicates that an object can be cloned for use in an edit buffer (see<see cref = "EditBuffer" />).
/// </summary>
/// <typeparam name="TSelf">
/// Type for which an edit buffer can be created.
/// </typeparam>
public interface IEditBufferCloneSource<TSelf> : IEditBufferCloneSource
    where TSelf : IEditBufferCloneSource<TSelf>
{
    new TSelf CreateEditBufferClone();

    IEditBufferCloneSource IEditBufferCloneSource.CreateEditBufferClone() => CreateEditBufferClone();
}