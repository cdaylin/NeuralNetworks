using System.Collections;
using System.Collections.Specialized;
using System.ComponentModel;

namespace Daylin.Utilities.Collections.Observable.Dictionaries;

/// <summary>
/// Represents a dictionary that raises change notifications when its contents are modified.
/// </summary>
/// <typeparam name="TKey">Type of dictionary key.</typeparam>
/// <typeparam name="TValue">Type of dictionary value.</typeparam>
public sealed class ObservableDictionary<TKey, TValue> :
    IDictionary<TKey, TValue>, IReadOnlyObservableDictionary<TKey, TValue>
    where TKey : notnull
{
    #region Construction

    public ObservableDictionary()
    {
        dictionary = [];
    }

    public ObservableDictionary(IDictionary<TKey, TValue> source)
    {
        source.ThrowIfNull();
        dictionary = new Dictionary<TKey, TValue>(source);
    }

    #endregion

    #region Public

    public event NotifyCollectionChangedEventHandler? CollectionChanged;

    public event PropertyChangedEventHandler? PropertyChanged;

    public ICollection<TKey> Keys => dictionary.Keys;

    public ICollection<TValue> Values => dictionary.Values;

    IEnumerable<TKey> IReadOnlyDictionary<TKey, TValue>.Keys => Keys;

    IEnumerable<TValue> IReadOnlyDictionary<TKey, TValue>.Values => Values;

    public int Count => dictionary.Count;

    public bool IsReadOnly => false;

    public TValue this[TKey key]
    {
        get => dictionary[key];

        set
        {
            if (dictionary.TryGetValue(key, out TValue? oldValue))
            {
                if (EqualityComparer<TValue>.Default.Equals(oldValue, value))
                    return;

                dictionary[key] = value;
                RaiseReplace(key, oldValue, value);
            }
            else
            {
                dictionary[key] = value;
                RaiseAdd(key, value);
            }
        }
    }

    public void Add(TKey key, TValue value)
    {
        dictionary.Add(key, value);
        RaiseAdd(key, value);
    }

    public bool Remove(TKey key)
    {
        if (dictionary.TryGetValue(key, out TValue? value) && dictionary.Remove(key))
        {
            RaiseRemove(key, value);
            return true;
        }

        return false;
    }

    public void Clear()
    {
        if (dictionary.Count == 0)
            return;

        dictionary.Clear();
        RaiseReset();
    }

    public bool ContainsKey(TKey key)
    {
        return dictionary.ContainsKey(key);
    }

    public bool TryGetValue(TKey key, out TValue value)
    {
        return dictionary.TryGetValue(key, out value!);
    }

    public void Add(KeyValuePair<TKey, TValue> item)
    {
        Add(item.Key, item.Value);
    }

    public bool Contains(KeyValuePair<TKey, TValue> item)
    {
        return dictionary.Contains(item);
    }

    public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex)
    {
        ((IDictionary<TKey, TValue>)dictionary).CopyTo(array, arrayIndex);
    }

    public bool Remove(KeyValuePair<TKey, TValue> item)
    {
        return Remove(item.Key);
    }

    public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
    {
        return dictionary.GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    #endregion

    #region Private

    private readonly Dictionary<TKey, TValue> dictionary;

    private void RaiseAdd(TKey key, TValue value)
    {
        CollectionChanged?.Invoke(
            this,
            new NotifyCollectionChangedEventArgs(
                NotifyCollectionChangedAction.Add,
                new KeyValuePair<TKey, TValue>(key, value)));

        RaiseCountAndIndexer();
    }

    private void RaiseRemove(TKey key, TValue value)
    {
        CollectionChanged?.Invoke(
            this,
            new NotifyCollectionChangedEventArgs(
                NotifyCollectionChangedAction.Remove,
                new KeyValuePair<TKey, TValue>(key, value)));

        RaiseCountAndIndexer();
    }

    private void RaiseReplace(TKey key, TValue oldValue, TValue newValue)
    {
        CollectionChanged?.Invoke(
            this,
            new NotifyCollectionChangedEventArgs(
                NotifyCollectionChangedAction.Replace,
                new KeyValuePair<TKey, TValue>(key, newValue),
                new KeyValuePair<TKey, TValue>(key, oldValue)));

        RaiseIndexer();
    }

    private void RaiseReset()
    {
        CollectionChanged?.Invoke(
            this,
            new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Reset));

        RaiseCountAndIndexer();
    }

    private void RaiseCountAndIndexer()
    {
        RaisePropertyChanged(nameof(Count));
        RaiseIndexer();
    }

    private void RaiseIndexer()
    {
        RaisePropertyChanged("Item[]");
        RaisePropertyChanged(nameof(Keys));
        RaisePropertyChanged(nameof(Values));
    }

    private void RaisePropertyChanged(string propertyName)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }

    #endregion
}
