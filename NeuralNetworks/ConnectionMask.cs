using Daylin.Utilities.Cloning;

using System.Diagnostics;
using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks;

/// <summary>
/// Responsible for tracking whether incoming connections to a neuron are active or masked.
/// </summary>
/// <remarks>
/// A 'masked' connection can be thought of as inactive or effectively nonexistent.  The value that a neuron
/// receives through a masked connection is always zero.
/// </remarks>
internal class ConnectionMask : CloneableObject, IConnectionMask
{
    #region Construction

    public ConnectionMask(int inputCount, int outputCount)
    {
        inputCount.ThrowIfNotPositive();
        outputCount.ThrowIfNotPositive();

        InputCount = inputCount;
        OutputCount = outputCount;

        MaskMatrix = new bool[OutputCount][];
    }

    protected ConnectionMask(ConnectionMask source)
    {
        InputCount = source.InputCount;
        OutputCount = source.OutputCount;

        MaskMatrix = new bool[OutputCount][];

        for (int index = 0; index < OutputCount; index++)
        {
            if (source.MaskMatrix[index] is not null)
                MaskMatrix[index] = (bool[])source.MaskMatrix[index]!.Clone();
        }
    }

    [JsonConstructor]
    protected ConnectionMask(
        int inputCount,
        int outputCount,
        bool[]?[] maskMatrix)
    {
        InputCount = inputCount;
        OutputCount = outputCount;
        MaskMatrix = maskMatrix;
    }

    #endregion

    #region Public

    /// <summary>
    /// Number of incoming connections (both masked and unmasked) for each neuron in the layer.
    /// </summary>
    public int InputCount { get; }

    /// <summary>
    /// Number of neurons in the layer.
    /// </summary>
    public int OutputCount { get; }

    public int GetMaskedConnectionCount(int outputIndex)
    {
        ValidateOutputIndex(outputIndex);

        if (MaskMatrix[outputIndex] is null)
            return 0;

        return MaskMatrix[outputIndex]!
            .Count(isMasked => isMasked);
    }

    public int GetUnmaskedConnectionCount(int outputIndex)
    {
        return InputCount - GetMaskedConnectionCount(outputIndex);
    }

    /// <summary>
    /// Returns a list of Boolean values indicating which connections are masked for a specified neuron.
    /// </summary>
    /// <param name="outputIndex">
    /// Output value index.
    /// </param>
    /// <returns>
    /// Returns a list of Boolean values, with one value for each incoming connection to the neuron.  A value of
    /// <c>true</c> indicates that the corresponding connection is masked.
    /// </returns>
    public IReadOnlyList<bool> GetConnectionMaskVector(int outputIndex)
    {
        ValidateOutputIndex(outputIndex);

        return MaskMatrix[outputIndex] ?? EmptyConnectionMaskVector;
    }

    /// <summary>
    /// Indicates whether an incoming connection to a neuron is masked.
    /// </summary>
    /// <param name="inputIndex">
    /// Input value index.
    /// </param>
    /// <param name="outputIndex">
    /// Output value index.
    /// </param>
    /// <returns>
    /// Returns <c>true</c> if and only if the connection is masked.
    /// </returns>
    public bool IsConnectionMasked(int inputIndex, int outputIndex)
    {
        ValidateInputIndex(inputIndex);
        ValidateOutputIndex(outputIndex);

        return GetConnectionMaskVector(outputIndex)[inputIndex];
    }

    public void SetIsConnectionMasked(int inputIndex, int outputIndex, bool isMasked)
    {
        ValidateInputIndex(inputIndex);
        ValidateOutputIndex(outputIndex);

        // initialize the mask vector, if needed
        if (MaskMatrix[outputIndex] is null)
        {
            if (!isMasked)
                return;
            else
                MaskMatrix[outputIndex] = new bool[InputCount];
        }

        // set the mask value for the connection
        MaskMatrix[outputIndex]![inputIndex] = isMasked;

        // if all connections are now unmasked, release the unused array
        if (!isMasked && !MaskMatrix[outputIndex]!.Any(value => value is true))
            MaskMatrix[outputIndex] = null!;
    }

    #endregion

    #region Protected

    protected override object CreateClone() => new ConnectionMask(this);

    #endregion

    #region Private

    [JsonInclude]
    private bool[]?[] MaskMatrix { get; }

    /// <summary>
    /// A mask vector that contains all <c>false</c> values.
    /// </summary>
    /// <remarks>
    /// This mask vector is shared by all neurons that have no masked connections, for efficiency.
    /// </remarks>
    private bool[] EmptyConnectionMaskVector
    {
        get
        {
            field ??= new bool[InputCount];

            return field;
        }
    } = null!;

    [StackTraceHidden]
    private void ValidateInputIndex(int inputIndex)
    {
        inputIndex.ThrowIfOutOfRange(0, InputCount - 1);
    }

    [StackTraceHidden]
    private void ValidateOutputIndex(int outputIndex)
    {
        outputIndex.ThrowIfOutOfRange(0, OutputCount - 1);
    }

    #endregion
}
