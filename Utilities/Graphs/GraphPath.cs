using System.Numerics;

namespace Daylin.Utilities.Graphs;

public class GraphPath<TNode, TCost> : IGraphPath<TNode, TCost>
    where TNode : notnull
    where TCost : INumber<TCost>
{
    /// <summary>
    /// Constructs an empty path comprising a single node and no edges.
    /// </summary>
    /// <param name="node">
    /// The start and end node path.
    /// </param>
    public GraphPath(TNode node)
    {
        node.ThrowIfNull();

        Edges = Enumerable.Empty<GraphEdge<TNode, TCost>>();

        StartNode = node;
        EndNode = node;
    }

    /// <summary>
    /// Constructs a path comprising one or more edges.
    /// </summary>
    /// <param name="edges">
    /// Sequence of edges constituting a path through a graph.  The end node of each edge should be the same as the
    /// start node of the following edge (except for the final edge).
    /// </param>
    public GraphPath(IEnumerable<GraphEdge<TNode, TCost>> edges)
    {
        edges.ThrowIfNull();
        edges.ThrowIfContainsNull();
        edges.ThrowIfEmpty();

        Edges = edges.ToArray();

        StartNode = Edges.First().StartNode;
        EndNode = Edges.Last().EndNode;
    }

    public TNode StartNode { get; }

    public TNode EndNode { get; }

    public TCost Cost => Edges.Select(edge => edge.Cost).Sum();

    public IEnumerable<GraphEdge<TNode, TCost>> Edges { get; }
}
