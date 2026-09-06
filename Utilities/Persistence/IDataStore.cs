namespace Daylin.Utilities.Persistence;

/// <summary>
/// Responsible for storing and retrieving data entities.
/// </summary>
/// <typeparam name="T">
/// Data entity type.
/// </typeparam>
public interface IDataStore<T> where T : notnull
{
    Task<bool> ContainsAsync(string key, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets an entity by key, or <c>null</c> if not found.
    /// </summary>
    Task<T?> GetAsync(string key, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets all entities from the store.
    /// </summary>
    /// <param name="exceptionHandler">
    /// Optional handler for exceptions that occur while reading individual entities. If provided, the handler is
    /// called and the operation continues with the next entity. If null, exceptions are thrown immediately. If
    /// the handler itself throws, an <see cref="Exceptions.ExceptionHandlingException"/> is thrown.
    /// </param>
    /// <param name="cancellationToken">
    /// Cancellation token.
    /// </param>
    /// <returns>
    /// All successfully loaded entities.
    /// </returns>
    Task<IReadOnlyList<T>> GetAllAsync(
        Action<Exception>? exceptionHandler = null,
        CancellationToken cancellationToken = default);

    Task PutAsync(string key, T data, CancellationToken cancellationToken = default);

    Task RemoveAsync(string key, CancellationToken cancellationToken = default);
}