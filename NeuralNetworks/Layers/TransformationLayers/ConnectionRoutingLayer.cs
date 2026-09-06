using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Layers.TransformationLayers;

internal sealed class ConnectionRoutingLayer : INetworkLayer
{
    #region Construction

    [JsonConstructor]
    public ConnectionRoutingLayer(IndexMap connectionMap)
    {
        connectionMap.ThrowIfNull();

        ConnectionMap = connectionMap;
    }

    #endregion

    #region Public

    public string ShortDescription => "Connection map";

    public int InputCount => ConnectionMap.Length;

    public int OutputCount => ConnectionMap.Length;

    public IEnumerable<int> Initialize(IEnumerable<int> outputConnectionCounts)
    {
        return ConnectionMap.MapBackward(outputConnectionCounts);
    }

    public Task<IEnumerable<float>> CalculateOutputAsync(
        IEnumerable<float> input,
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(ConnectionMap.MapForward(input));
    }

    public async Task<(IEnumerable<float>, BackpropagateErrorsAsync)> FeedForwardAsync(
        IEnumerable<float> input,
        BackpropagateErrorsAsync? backpropagateErrorsAsync,
        CancellationToken cancellationToken = default)
    {
        return (await CalculateOutputAsync(input), BackpropagateErrorsAsync);

        async Task BackpropagateErrorsAsync(
            float learningRate,
            IEnumerable<float> errors,
            CancellationToken cancellationToken)
        {
            if (backpropagateErrorsAsync is null)
                return;

            await backpropagateErrorsAsync(
                learningRate,
                ConnectionMap.MapBackward(errors),
                cancellationToken);
        }
    }

    #endregion

    #region Private

    [JsonInclude]
    private IndexMap ConnectionMap { get; }

    #endregion
}
