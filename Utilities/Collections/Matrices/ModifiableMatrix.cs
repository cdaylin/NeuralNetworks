namespace Daylin.Utilities.Collections.Matrices;

public class ModifiableMatrix<T> : Matrix<T>, IModifiableMatrix<T>
{
    public ModifiableMatrix(int rowCount, int columnCount)
        : base(rowCount, columnCount)
    {
    }

    protected ModifiableMatrix(Matrix<T> source)
        : base(source)
    {
    }

    public new ModifiableMatrix<T> Clone()
    {
        return (ModifiableMatrix<T>)base.Clone();
    }

    public new T this[int rowIndex, int columnIndex]
    {
        get => base[rowIndex, columnIndex];

        set => this[GetArrayIndex(rowIndex, columnIndex)] = value;
    }

    public void SetAt(int rowIndex, int columnIndex, T value)
    {
        this[rowIndex, columnIndex] = value;
    }

    protected new T this[int index]
    {
        get => base[index];

        set => data[index] = value;
    }

    protected override object CreateClone()
    {
        return new ModifiableMatrix<T>(this);
    }
}
