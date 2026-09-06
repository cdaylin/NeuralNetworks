using System.Numerics;

namespace Daylin.Utilities.Graphs;

public class GraphEdge<TNode, TCost>
    where TNode : notnull
    where TCost : INumber<TCost>
{
    public GraphEdge(TNode startNode, TNode endNode, TCost cost)
    {
        cost.ThrowIfNegative();

        StartNode = startNode;
        EndNode = endNode;
        Cost = cost;
    }

    public TNode StartNode { get; }

    public TNode EndNode { get; }

    public TCost Cost { get; }

    public override string ToString()
    {
        return $"({StartNode} -> {EndNode})";
    }
}
