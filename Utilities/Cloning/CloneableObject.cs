namespace Daylin.Utilities.Cloning;

/// <summary>
/// Abstract base class that provides a partial implementation of <see cref="ICloneable"/>.
/// </summary>
public abstract class CloneableObject : ICloneable
{
    protected CloneableObject()
    {
    }

    protected CloneableObject(CloneableObject source)
    {
        // This copy constructor is not strictly necessary, because this base class does not need to copy
        //  any data.  It is provided in order to allow for a consistent pattern: each derived class should
        //  implement a copy constructor that calls the copy constructor of the base class.
    }

    public CloneableObject Clone()
    {
        return (CloneableObject)((ICloneable)this).Clone();
    }

    object ICloneable.Clone()
    {
        object clone = CreateClone();

        if (!clone.GetType().Equals(GetType()))
        {
            throw new Exception(
                $"Incorrect implementation of {typeof(ICloneable)}.  " +
                $"The cloned object is not of the same concrete type as the source object.  " +
                $"Source type: '{GetType().FullName}'.  Clone type: '{clone.GetType().FullName}'.");
        }

        return clone;
    }

    /// <summary>
    /// Constructs and returns a new object that is a clone of this object.
    /// </summary>
    /// <remarks>
    /// Each concrete derived class should override this method to call its own copy constructor.  The copy
    /// constructor should be implemented to call the copy constructor of its base class and then copy any data
    /// managed by the derived class.
    /// </remarks>
    /// <returns>
    /// A new clone of this object.
    /// </returns>
    protected abstract object CreateClone();
}