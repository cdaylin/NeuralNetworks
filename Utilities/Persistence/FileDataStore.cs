using System.Collections.Concurrent;

using Daylin.Utilities.Exceptions;
using Daylin.Utilities.Threading;

namespace Daylin.Utilities.Persistence;

/// <summary>
/// Data store that persists string content as individual files in a directory.
/// </summary>
/// <remarks>
/// This store is format-agnostic; it only handles file I/O.
/// Serialization format (JSON, XML, etc.) is the responsibility of the codec.
/// </remarks>
public sealed class FileDataStore : IDataStore<string>
{
    #region Construction

    /// <summary>
    /// Creates a new file data store.
    /// </summary>
    /// <param name="directoryPath">
    /// Path to the directory where files are stored.
    /// </param>
    /// <param name="fileExtension">
    /// File extension including the leading dot (e.g., ".json", ".xml").
    /// </param>
    public FileDataStore(string directoryPath, string fileExtension)
    {
        fileExtension.ThrowIfNull();

        DirectoryPath = directoryPath;  // Setter validates and ensures directory exists
        FileExtension = fileExtension;
    }

    #endregion

    #region Public

    /// <summary>
    /// Directory path where files are stored.
    /// </summary>
    public string DirectoryPath
    {
        get;

        set
        {
            value.ThrowIfNull();
            field = value;
            EnsureDirectoryExists();
        }
    }

    /// <summary>
    /// File extension used for stored files.
    /// </summary>
    public string FileExtension { get; }

    public Task<bool> ContainsAsync(string key, CancellationToken cancellationToken = default)
    {
        key.ThrowIfNull();

        string filePath = GetFilePath(key);

        return Task.FromResult(File.Exists(filePath));
    }

    public async Task<IReadOnlyList<string>> GetAllAsync(
        Action<Exception>? exceptionHandler = null,
        CancellationToken cancellationToken = default)
    {
        if (!Directory.Exists(DirectoryPath))
            return [];

        List<string> contents = [];

        string searchPattern = "*" + FileExtension;

        foreach (string filePath in Directory.EnumerateFiles(DirectoryPath, searchPattern))
        {
            cancellationToken.ThrowIfCancellationRequested();

            try
            {
                string content = await File.ReadAllTextAsync(filePath, cancellationToken);
                contents.Add(content);
            }
            catch (Exception exception) when (IsRecoverableException(exception))
            {
                HandleException(exception, exceptionHandler);
            }
        }

        return contents;
    }

    public async Task PutAsync(string key, string data, CancellationToken cancellationToken = default)
    {
        key.ThrowIfNull();
        data.ThrowIfNull();

        using (await GetLock(key).LockAsync(cancellationToken))
        {
            EnsureDirectoryExists();

            await File.WriteAllTextAsync(GetFilePath(key), data, cancellationToken);
        }
    }

    public async Task RemoveAsync(string key, CancellationToken cancellationToken = default)
    {
        key.ThrowIfNull();

        using (await GetLock(key).LockAsync(cancellationToken))
        {
            string filePath = GetFilePath(key);

            if (File.Exists(filePath))
                File.Delete(filePath);
        }
    }

    public async Task<string?> GetAsync(string key, CancellationToken cancellationToken = default)
    {
        key.ThrowIfNull();

        string filePath = GetFilePath(key);

        if (!File.Exists(filePath))
            return null;

        return await File.ReadAllTextAsync(filePath, cancellationToken);
    }

    #endregion

    #region Private

    /// <summary>
    /// Per-key locks to prevent concurrent write operations on the same file.
    /// </summary>
    private ConcurrentDictionary<string, SemaphoreSlim> WriteLocks { get; } = [];

    private SemaphoreSlim GetLock(string key)
    {
        return WriteLocks.GetOrAdd(key, _ => new SemaphoreSlim(1, 1));
    }

    private string GetFilePath(string key)
    {
        return Path.Combine(DirectoryPath, key + FileExtension);
    }

    private void EnsureDirectoryExists()
    {
        if (!Directory.Exists(DirectoryPath))
            Directory.CreateDirectory(DirectoryPath);
    }

    /// <summary>
    /// Determines whether an exception is recoverable (skip the item and continue) or should fail the entire
    /// operation.
    /// </summary>
    /// <remarks>
    /// Recoverable: file-specific I/O errors (locked, corrupted).
    /// Non-recoverable: access denied, directory not found, out of memory - these affect the entire operation.
    /// </remarks>
    private static bool IsRecoverableException(Exception exception)
    {
        return exception is IOException
            and not DirectoryNotFoundException
            and not DriveNotFoundException
            and not PathTooLongException;
    }

    private static void HandleException(Exception exception, Action<Exception>? exceptionHandler)
    {
        if (exceptionHandler is null)
            throw exception;

        try
        {
            exceptionHandler(exception);
        }
        catch (Exception handlerException)
        {
            throw new ExceptionHandlingException(exception, handlerException);
        }
    }

    #endregion
}
