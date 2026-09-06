using System.Numerics;

namespace Daylin.Utilities.Graphs;

public abstract class GraphView<TNode, TCost> : IGraph<TNode, TCost>
    where TNode : notnull
    where TCost : INumber<TCost>
{
    public GraphView(IGraph<TNode, TCost> graph)
    {
        graph.ThrowIfNull();

        Graph = graph;
    }

    public IEnumerable<TNode> Nodes => Graph.Nodes;

    public bool Contains(TNode node)
    {
        return Graph.Contains(node);
    }

    public abstract IEnumerable<GraphEdge<TNode, TCost>> GetEdges(TNode node);

    protected IGraph<TNode, TCost> Graph { get; }
}
