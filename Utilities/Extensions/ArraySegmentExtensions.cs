namespace Daylin.Utilities.Extensions;

/// <summary>
/// Extension methods for struct <see cref="ArraySegment{T}"/>.
/// </summary>
public static class ArraySegmentExtensions
{
    public static ArraySegment<T> GetSegment<T>(this ArraySegment<T> source, int offset, int length)
    {
        return new ArraySegment<T>(source.Array!, offset + source.Offset, length + source.Count);
    }
}
