using System.Numerics;

namespace Daylin.Utilities.Graphs;

public interface IGraph<TNode, TCost>
    where TNode : notnull
    where TCost : INumber<TCost>
{
    IEnumerable<TNode> Nodes { get; }

    bool Contains(TNode node);

    IEnumerable<GraphEdge<TNode, TCost>> GetEdges(TNode node);
}
