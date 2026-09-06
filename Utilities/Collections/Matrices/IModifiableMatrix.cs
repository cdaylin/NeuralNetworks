namespace Daylin.Utilities.Collections.Matrices;

public interface IModifiableMatrix<T> : IMatrix<T>
{
    new T this[int rowIndex, int columnIndex] { get; set; }

    void SetAt(int rowIndex, int columnIndex, T value);
}
