using System.Numerics;

namespace Daylin.Utilities.Graphs;

public interface IGraphPath<TNode, TCost>
    where TNode : notnull
    where TCost : INumber<TCost>
{
    TNode StartNode { get; }

    TNode EndNode { get; }

    TCost Cost { get; }

    IEnumerable<GraphEdge<TNode, TCost>> Edges { get; }
}
