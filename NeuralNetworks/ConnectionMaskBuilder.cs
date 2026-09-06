namespace Daylin.NeuralNetworks;

public class ConnectionMaskBuilder
{
    #region Construction

    public ConnectionMaskBuilder(int inputCount, int outputCount)
    {
        ConnectionMask = new ConnectionMask(inputCount, outputCount);
    }

    #endregion

    #region Public

    public int InputCount => ConnectionMask.InputCount;

    public int OutputCount => ConnectionMask.OutputCount;

    public void MaskAllConnections(int neuronIndex)
    {
        for (int connectionIndex = 0; connectionIndex < ConnectionMask.InputCount; connectionIndex++)
            SetIsConnectionMasked(neuronIndex, connectionIndex, isMasked: true);
    }

    public void SetIsConnectionMasked(int neuronIndex, int connectionIndex, bool isMasked)
    {
        ConnectionMask.SetIsConnectionMasked(connectionIndex, neuronIndex, isMasked);
    }

    public IConnectionMask Build() => (IConnectionMask)ConnectionMask.Clone();

    #endregion

    #region Private

    private ConnectionMask ConnectionMask { get; }

    #endregion
}
