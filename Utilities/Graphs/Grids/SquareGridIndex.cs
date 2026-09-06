namespace Daylin.Utilities.Graphs.Grids;

/// <summary>
/// A two-dimensional index for identifying a node in a two-dimensional graph comprising squares arranged in
/// rows and columns.
/// </summary>
public readonly struct SquareGridIndex : IEquatable<SquareGridIndex>
{
    public static bool operator ==(SquareGridIndex left, SquareGridIndex right)
    {
        return Equals(left, right);
    }

    public static bool operator !=(SquareGridIndex left, SquareGridIndex right)
    {
        return !(left == right);
    }

    public static int CalculateDistance(SquareGridIndex left, SquareGridIndex right)
    {
        left.ThrowIfNull();
        right.ThrowIfNull();

        return Math.Abs(left.RowIndex - right.RowIndex)
            + Math.Abs(left.ColumnIndex - right.ColumnIndex);
    }

    public SquareGridIndex() : this(0, 0)
    {
    }

    public SquareGridIndex(int rowIndex, int columnIndex)
    {
        rowIndex.ThrowIfNegative();
        columnIndex.ThrowIfNegative();

        RowIndex = rowIndex;
        ColumnIndex = columnIndex;
    }

    public int RowIndex { get; }

    public int ColumnIndex { get; }

    public bool Equals(SquareGridIndex other)
    {
        return RowIndex.Equals(other.RowIndex) && ColumnIndex.Equals(other.ColumnIndex);
    }

    public override bool Equals(object? other)
    {
        return other is SquareGridIndex otherGridIndex && Equals(otherGridIndex);
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(RowIndex, ColumnIndex);
    }

    public override string ToString()
    {
        return $"({RowIndex}, {ColumnIndex})";
    }
}
