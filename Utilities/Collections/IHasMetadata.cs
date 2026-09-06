using Daylin.Utilities.Collections.Observable.Dictionaries;

namespace Daylin.Utilities.Collections;

/// <summary>
/// Represents an object that contains a collection of metadata key/value pairs.
/// </summary>
public interface IHasMetadata
{
    /// <summary>
    /// Metadata key/value pairs associated with the object.
    /// </summary>
    ObservableDictionary<string, string> Metadata { get; }
}