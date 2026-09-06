namespace Daylin.Utilities.Collections.Dictionaries;

public interface IMultiValueDictionary<TKey, TValue>
    : IReadableMultiValueDictionary<TKey, TValue>
    where TKey : notnull
{
    /// <summary>
    /// Adds a value mapped to a key.
    /// </summary>
    /// <remarks>
    /// The only guaranteed postcondition for this operation is that <paramref name="value"/> is contained in the
    /// collection mapped to <paramref name="key"/>.  Whether the value count increases depends in part upon the
    /// type of collection used to store the values.  For a <see cref="IList{T}">List</see>, the value is always
    /// added and the count increases by one.  For a <see cref="ISet{T}">Set</see>, the value is only added if
    /// it does not already exist in the collection.
    /// </remarks>
    /// <param name="key">
    /// Distinct identifier for a collection of values.
    /// </param>
    /// <param name="value">
    /// The value to add to the collection mapped to <paramref name="key"/>.
    /// </param>
    void Add(TKey key, TValue value);

    /// <summary>
    /// Adds values mapped to a key.
    /// </summary>
    /// <remarks>
    /// The only guaranteed postcondition for this operation is that each value enumerated by
    /// <paramref name="values"/> is contained in the collection mapped to <paramref name="key"/>.  Whether the 
    /// value count increases depends in part upon the type of collection used to store the values.  For a
    /// <see cref="IList{T}">List</see>, each value is always added and the count increases by one.  For a
    /// <see cref="ISet{T}">Set</see>, each value is only added if it does not already exist in the collection.
    /// </remarks>
    /// <param name="key">
    /// Distinct identifier for a collection of values.
    /// </param>
    /// <param name="values">
    /// Zero-to-many values to add to the collection mapped to <paramref name="key"/>.
    /// </param>
    void AddRange(TKey key, params IEnumerable<TValue> values);

    /// <summary>
    /// Removes a value mapped to a key.
    /// </summary>
    /// <remarks>
    /// If <paramref name="value"/> is contained in the collection mapped to <paramref name="key"/>, one instance
    /// of <paramref name="value"/> is removed and the value count is decreased by one.  If <paramref name="value"/>
    /// is not contained in the collection, then this operation has no effect.
    /// </remarks>
    /// <param name="key">
    /// Distinct identifier for a collection of values.
    /// </param>
    /// <param name="value">
    /// The value to add to the collection mapped to <paramref name="key"/>.
    /// </param>
    /// <returns>
    /// Returns <c>true</c> if and only if <paramref name="value"/> is removed from the collection of values mapped
    /// to <paramref name="key"/>.
    /// </returns>
    bool Remove(TKey key, TValue value);

    bool RemoveKey(TKey key);

    void Clear();
}
