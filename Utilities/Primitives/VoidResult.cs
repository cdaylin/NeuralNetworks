namespace Daylin.Utilities.Primitives;

/// <summary>
/// When used as a generic argument, acts as a placeholder for a return type of <c>void</c>.
/// </summary>
public readonly struct VoidResult
{
    public static readonly VoidResult Instance = new();
}
