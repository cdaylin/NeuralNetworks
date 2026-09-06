namespace Daylin.Utilities.FileSystem;

/// <summary>
/// Provides helper methods for working with folder and file paths.
/// </summary>
/// <seealso cref="Path"/>
public static class PathUtilities
{
    /// <summary>
    /// Converts a relative path to its canonical form by resolving any <c>.</c> or <c>..</c> segments.
    /// </summary>
    /// <param name="basePath">
    /// A fully qualified base path.
    /// </param>
    /// <param name="relativePath">
    /// A relative path to canonicalize.
    /// </param>
    /// <returns>
    /// The canonical relative path, or an empty string if the path resolves to the base path itself.
    /// </returns>
    public static string CanonicalizeRelativePath(string basePath, string relativePath)
    {
        basePath.ThrowIfNull();
        relativePath.ThrowIfNull();

        if (!Path.IsPathFullyQualified(basePath))
            throw new ArgumentException("The base path must be fully qualified.", nameof(basePath));

        if (Path.IsPathFullyQualified(relativePath))
            throw new ArgumentException("The relative path must not be fully qualified.", nameof(relativePath));

        if (Path.IsPathRooted(relativePath))
            throw new ArgumentException("The relative path must not be rooted.", nameof(relativePath));

        string canonicalAbsolutePath = Path.GetFullPath(relativePath, basePath);

        string canonicalRelativePath = Path.GetRelativePath(basePath, canonicalAbsolutePath);

        return canonicalRelativePath == "." ? string.Empty : canonicalRelativePath;
    }

    /// <summary>
    /// Splits a path into its component segments.
    /// </summary>
    /// <param name="path">
    /// A file or folder path.
    /// </param>
    /// <returns>
    /// An array of path segments, excluding empty entries.
    /// </returns>
    public static string[] GetSegments(string path)
    {
        path.ThrowIfNull();

        return path.Split(
            [Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar],
            StringSplitOptions.RemoveEmptyEntries);
    }
}
