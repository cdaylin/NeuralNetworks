namespace Daylin.Utilities.Collections.Matrices;

public interface IMatrix<T> : IEnumerable<T>
{
    int RowCount { get; }

    int ColumnCount { get; }

    int Count => RowCount * ColumnCount;

    bool IsInBounds(int rowIndex, int columnIndex)
    {
        return rowIndex >= 0
            && rowIndex < RowCount
            && columnIndex >= 0
            && columnIndex < ColumnCount;
    }

    T this[int rowIndex, int columnIndex] { get; }

    T GetAt(int rowIndex, int columnIndex);
}
