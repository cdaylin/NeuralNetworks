using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks;

/// <summary>
/// Maps a sequence of indices to the same set of indices ordered differently.
/// </summary>
/// <remarks>
/// An <see cref="IndexMap"/> can be used to reorder a sequence according to a static mapping of indices.
/// </remarks>
public sealed class IndexMap
{
    #region Construction

    /// <summary>
    /// Creates an index map.
    /// </summary>
    /// <param name="mappedIndices">
    /// An sequence of distinct integers in the range <c>[0, mappedIndices.Count() - 1]</c>, in any order.  The
    /// position of each integer in the sequence will be mapped to the index represented by the integer.
    /// </param>
    public IndexMap(IEnumerable<int> mappedIndices)
    {
        mappedIndices.ThrowIfNull();
        mappedIndices.ThrowIfEmpty();

        ForwardIndices = mappedIndices.ToArray();

        if (!mappedIndices.AreContentsEqual(Enumerable.Range(0, ForwardIndices.Length)))
        {
            throw new ArgumentException($"Argument '{nameof(mappedIndices)}' "
                + $"must contain distinct integers in the range [0, {nameof(mappedIndices)}.Count() - 1].");
        }

        BackwardIndices = CreateBackwardIndices(ForwardIndices);
    }

    [JsonConstructor]
    [SuppressMessage("CodeQuality", "IDE0051:Remove unused private members",
        Justification = "Called via reflection")]
    private IndexMap(int[] forwardIndices)
    {
        ForwardIndices = forwardIndices;
        BackwardIndices = CreateBackwardIndices(forwardIndices);
    }

    #endregion

    #region Public

    [JsonIgnore]
    public int Length => ForwardIndices.Length;

    public int MapForward(int inputIndex)
    {
        inputIndex.ThrowIfOutOfRange(0, Length - 1);

        return ForwardIndices[inputIndex];
    }

    public int MapBackward(int outputIndex)
    {
        outputIndex.ThrowIfOutOfRange(0, Length - 1);

        return BackwardIndices[outputIndex];
    }

    public IEnumerable<T> MapForward<T>(IEnumerable<T> values)
    {
        values.ThrowIfNull();

        DebugOnlyValidateValuesCount(values);

        T[] result = new T[Length];

        int index = 0;

        foreach (T value in values)
            result[MapForward(index++)] = value;

        return result;
    }

    public IReadOnlyList<T> MapBackward<T>(IEnumerable<T> values)
    {
        values.ThrowIfNull();

        DebugOnlyValidateValuesCount(values);

        T[] result = new T[Length];

        int index = 0;

        foreach (T value in values)
            result[MapBackward(index++)] = value;

        return result;
    }

    #endregion

    #region Private

    [StackTraceHidden]
    [Conditional("DEBUG")]
    private void DebugOnlyValidateValuesCount<T>(IEnumerable<T> values)
    {
        // this can significantly hurt performance, so suppress it in release builds
        values.Count().ThrowIfNotEqualTo(Length);
    }

    [JsonInclude]
    private int[] ForwardIndices { get; }

    private int[] BackwardIndices { get; }

    private static int[] CreateBackwardIndices(int[] forwardIndices)
    {
        int[] backwardIndices = new int[forwardIndices.Length];

        for (int index = 0; index < forwardIndices.Length; index++)
            backwardIndices[forwardIndices[index]] = index;

        return backwardIndices;
    }

    #endregion
}
