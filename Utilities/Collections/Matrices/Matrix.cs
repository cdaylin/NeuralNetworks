using Daylin.Utilities.Cloning;

using System.Collections;

namespace Daylin.Utilities.Collections.Matrices;

public abstract class Matrix<T> : CloneableObject, IMatrix<T>
{
    protected Matrix(int rowCount, int columnCount)
    {
        rowCount.ThrowIfNotPositive();
        columnCount.ThrowIfNotPositive();

        RowCount = rowCount;
        ColumnCount = columnCount;

        data = new T[rowCount * columnCount];
    }

    protected Matrix(Matrix<T> source)
    {
        source.ThrowIfNull();

        RowCount = source.RowCount;
        ColumnCount = source.ColumnCount;

        data = new T[source.Count];

        source.data.CopyTo(data, 0);
    }

    public new Matrix<T> Clone()
    {
        return (Matrix<T>)base.Clone();
    }

    public IEnumerator<T> GetEnumerator()
    {
        return ((IEnumerable<T>)data).GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return data.GetEnumerator();
    }

    public int RowCount { get; }

    public int ColumnCount { get; }

    public int Count => RowCount * ColumnCount;

    public bool IsInBounds(int rowIndex, int columnIndex)
    {
        return rowIndex >= 0 && rowIndex < RowCount && columnIndex >= 0 && columnIndex < ColumnCount;
    }

    public T this[int rowIndex, int columnIndex] => this[GetArrayIndex(rowIndex, columnIndex)];

    public T GetAt(int rowIndex, int columnIndex)
    {
        return this[rowIndex, columnIndex];
    }

    protected T this[int index] => data[index];

    protected int GetArrayIndex(int rowIndex, int columnIndex)
    {
        return rowIndex * ColumnCount + columnIndex;
    }

    protected readonly T[] data;
}
