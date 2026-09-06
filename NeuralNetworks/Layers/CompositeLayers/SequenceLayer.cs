using System.Text;
using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Layers.CompositeLayers;

/// <summary>
/// A composite network layer comprising two or more nested layers that are connected sequentially.
/// </summary>
/// <remarks>
/// The input to a sequence layer is the input to the first nested layer.  The output from a sequence layer is the
/// output from the last nested layer.  Nested layers are connected sequentially, in the order provided to the
/// constructor.  The output from each nested layer is used as the input for the subsequent layer.  
/// </remarks>
internal sealed class SequenceLayer : CompositeNetworkLayer
{
    [JsonConstructor]
    public SequenceLayer(IEnumerable<INetworkLayer> layers)
        : base(layers)
    {
    }

    public override string ToString() => ShortDescription;

    [JsonIgnore]
    public override string ShortDescription
    {
        get
        {
            StringBuilder description = new();

            description.Append(Layers.First().ShortDescription);

            foreach (INetworkLayer layer in Layers.Skip(1))
                description.Append($" -> {layer.ShortDescription}");

            return description.ToString();
        }
    }

    [JsonIgnore]
    public override int InputCount => Layers.First().InputCount;

    [JsonIgnore]
    public override int OutputCount => Layers.Last().OutputCount;

    public override IEnumerable<int> Initialize(IEnumerable<int> outgoingConnectionCounts)
    {
        IEnumerable<int> connectionCounts = outgoingConnectionCounts;

        foreach (INetworkLayer layer in Layers.Reverse())
            connectionCounts = layer.Initialize(connectionCounts);

        return connectionCounts;
    }

    public override async Task<IEnumerable<float>> CalculateOutputAsync(
        IEnumerable<float> input,
        CancellationToken cancellationToken = default)
    {
        IEnumerable<float>? values = input;

        foreach (INetworkLayer layer in Layers)
        {
            cancellationToken.ThrowIfCancellationRequested();

            values = await layer.CalculateOutputAsync(values, cancellationToken);
        }

        return values;
    }

    public override async Task<(IEnumerable<float>, BackpropagateErrorsAsync)> FeedForwardAsync(
        IEnumerable<float> input,
        BackpropagateErrorsAsync? backpropagateErrorsAsync,
        CancellationToken cancellationToken = default)
    {
        IEnumerable<float> values = input;

        foreach (INetworkLayer layer in Layers)
        {
            cancellationToken.ThrowIfCancellationRequested();

            (values, backpropagateErrorsAsync) = await layer.FeedForwardAsync(
                values, backpropagateErrorsAsync, cancellationToken);
        }

        return (values, backpropagateErrorsAsync!);
    }
}