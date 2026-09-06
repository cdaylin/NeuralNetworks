using System.Numerics;

namespace Daylin.Utilities.Graphs.Grids;

/// <summary>
/// A graph comprising a grid of squares.
/// </summary>
/// <typeparam name="TCost">
/// Number type used for specifying the cost for edges in the graph.
/// </typeparam>
public interface ISquareGrid<TCost> : IGraph<SquareGridIndex, TCost>
    where TCost : INumber<TCost>
{
}
