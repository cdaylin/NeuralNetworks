namespace Daylin.Utilities.Collections.Dictionaries;

/// <summary>
/// A dictionary that maps each key to zero or more values.
/// </summary>
/// <typeparam name="TKey">
/// Type of key.
/// </typeparam>
/// <typeparam name="TValue">
/// Type of value.
/// </typeparam>
/// <typeparam name="TCollection">
/// Type of collection used to store values for each key.
/// </typeparam>
public abstract class MultiValueDictionary<TKey, TValue, TCollection>
    : IMultiValueDictionary<TKey, TValue>
    where TKey : notnull
    where TCollection : ICollection<TValue>
{
    public MultiValueDictionary(
        IEqualityComparer<TKey>? keyComparer = null)
    {
        KeyComparer = keyComparer ?? EqualityComparer<TKey>.Default;

        Dictionary = new Dictionary<TKey, TCollection>(KeyComparer);
    }

    public IEqualityComparer<TKey> KeyComparer { get; }

    public int KeyCount => Dictionary.Count;

    public int GetValueCount(TKey key)
    {
        key.ThrowIfNull();

        return TryGetValuesCollection(key, out TCollection? values) ? values!.Count : 0;
    }

    public IEnumerable<TKey> Keys => Dictionary.Keys;

    public bool ContainsKey(TKey key)
    {
        key.ThrowIfNull();

        return Dictionary.ContainsKey(key);
    }

    public bool ContainsValue(TKey key, TValue value)
    {
        key.ThrowIfNull();

        return TryGetValuesCollection(key, out TCollection? values) && values!.Contains(value);
    }

    public IEnumerable<TValue> GetValues(TKey key)
    {
        key.ThrowIfNull();

        return TryGetValuesCollection(key, out TCollection? values) ? values! : Enumerable.Empty<TValue>();
    }

    public void Add(TKey key, TValue value)
    {
        key.ThrowIfNull();

        GetOrCreateValuesCollection(key).Add(value);
    }

    public virtual void AddRange(TKey key, params IEnumerable<TValue> values)
    {
        key.ThrowIfNull();
        values.ThrowIfNull();

        foreach (TValue value in values)
            Add(key, value);
    }

    public bool RemoveKey(TKey key)
    {
        key.ThrowIfNull();

        return Dictionary.Remove(key);
    }

    public bool Remove(TKey key, TValue value)
    {
        key.ThrowIfNull();

        if (!TryGetValuesCollection(key, out TCollection? values))
            return false;

        if (!values!.Remove(value))
            return false;

        if (values.Count == 0)
            Dictionary.Remove(key);

        return true;
    }

    public void Clear()
    {
        Dictionary.Clear();
    }

    protected bool TryGetValuesCollection(TKey key, out TCollection? values)
    {
        return Dictionary.TryGetValue(key, out values);
    }

    protected TCollection GetOrCreateValuesCollection(TKey key)
    {
        key.ThrowIfNull();

        if (!TryGetValuesCollection(key, out TCollection? values))
        {
            values = CreateValuesCollection();
            Dictionary.Add(key, values);
        }

        return values!;
    }

    /// <summary>
    /// Constructs a collection object for containing values mapped to a key.
    /// </summary>
    /// <returns>
    /// Returns a new collection for storing values mapped to a key.
    /// </returns>
    protected abstract TCollection CreateValuesCollection();

    private Dictionary<TKey, TCollection> Dictionary { get; }
}
