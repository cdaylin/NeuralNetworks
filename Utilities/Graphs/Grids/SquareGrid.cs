using System.Numerics;

namespace Daylin.Utilities.Graphs.Grids;

public class SquareGrid<TCost> : ISquareGrid<TCost>
    where TCost : INumber<TCost>
{
    public SquareGrid(int rowCount, int columnCount)
    {
        rowCount.ThrowIfNotPositive();
        columnCount.ThrowIfNotPositive();

        RowCount = rowCount;
        ColumnCount = columnCount;
    }

    public int RowCount { get; }

    public int ColumnCount { get; }

    public virtual bool Contains(SquareGridIndex node)
    {
        return node.RowIndex >= 0
            && node.RowIndex < RowCount
            && node.ColumnIndex >= 0
            && node.ColumnIndex < ColumnCount;
    }

    public virtual IEnumerable<SquareGridIndex> Nodes
    {
        get
        {
            for (int row = 0; row < RowCount; row++)
                for (int column = 0; column < ColumnCount; column++)
                    yield return new SquareGridIndex(row, column);
        }
    }

    public virtual IEnumerable<GraphEdge<SquareGridIndex, TCost>> GetEdges(SquareGridIndex node)
    {
        return GetNeighbors(node)
            .Select(neighborIndex => new GraphEdge<SquareGridIndex, TCost>(node, neighborIndex, TCost.One));
    }

    public virtual IEnumerable<SquareGridIndex> GetNeighbors(SquareGridIndex node)
    {
        if (!Contains(node))
            yield break;

        if (node.RowIndex > 0)
            yield return new SquareGridIndex(node.RowIndex - 1, node.ColumnIndex);

        if (node.ColumnIndex > 0)
            yield return new SquareGridIndex(node.RowIndex, node.ColumnIndex - 1);

        if (node.RowIndex < RowCount - 1)
            yield return new SquareGridIndex(node.RowIndex + 1, node.ColumnIndex);

        if (node.ColumnIndex < ColumnCount - 1)
            yield return new SquareGridIndex(node.RowIndex, node.ColumnIndex + 1);
    }
}
