namespace Daylin.Utilities.Extensions;

/// <summary>
/// Extension methods for type <see cref="IDictionary{TKey, TValue}"/> and 
/// <see cref="IReadOnlyDictionary{TKey, TValue}"/>.
/// </summary>
public static class DictionaryExtensions
{
    public static TResult GetValue<TResult>(
        this IDictionary<string, object?> dictionary,
        string key,
        TResult defaultValue = default!)
    {
        if (!dictionary.TryGetValue(key, out object? value))
            return defaultValue;

        if (value is TResult result)
            return result;

        throw new InvalidCastException(
            $"Value for key '{key}' exists but is of type '{value?.GetType().FullName}', " +
                $"not assignable to '{typeof(TResult).FullName}'.");
    }
}
