using Daylin.TestUtilities;

namespace Daylin.NeuralNetworks;

/// <summary>
/// Tests the connection mask a sparse layer is built from.
/// </summary>
/// <remarks>
/// A masked connection is one a neuron does not have:  it carries no value forward and holds no weight.  The
/// builder names its arguments by neuron and connection, where the mask itself names them by input value and
/// output value, so which index is which is part of what these tests establish.
/// </remarks>
[TestClass]
public class ConnectionMaskTests
{
    #region Public

    [TestMethod]
    public void AConnectionIsUnmaskedUntilItIsMasked()
    {
        IConnectionMask mask = new ConnectionMaskBuilder(inputCount: 3, outputCount: 2).Build();

        Assert.IsFalse(mask.IsConnectionMasked(inputIndex: 2, outputIndex: 1));
        Assert.AreEqual(0, mask.GetMaskedConnectionCount(outputIndex: 1));
        Assert.AreEqual(3, mask.GetUnmaskedConnectionCount(outputIndex: 1));
    }

    [TestMethod]
    public void TheBuilderMasksTheConnectionBetweenTheNeuronAndTheInputValueItNames()
    {
        ConnectionMaskBuilder builder = new(inputCount: 3, outputCount: 2);

        builder.SetIsConnectionMasked(neuronIndex: 1, connectionIndex: 0, isMasked: true);

        IConnectionMask mask = builder.Build();

        // The neuron is the output value and the connection is the input value, so the masked connection is
        // (input 0, output 1) rather than (input 1, output 0).
        Assert.IsTrue(mask.IsConnectionMasked(inputIndex: 0, outputIndex: 1));
        Assert.IsFalse(mask.IsConnectionMasked(inputIndex: 1, outputIndex: 0));

        Assert.AreEqual(1, mask.GetMaskedConnectionCount(outputIndex: 1));
        Assert.AreEqual(2, mask.GetUnmaskedConnectionCount(outputIndex: 1));
        Assert.AreEqual(0, mask.GetMaskedConnectionCount(outputIndex: 0));
    }

    [TestMethod]
    public void MaskingEveryConnectionLeavesANeuronWithNone()
    {
        ConnectionMaskBuilder builder = new(inputCount: 3, outputCount: 2);

        builder.MaskAllConnections(neuronIndex: 0);

        IConnectionMask mask = builder.Build();

        Assert.AreEqual(3, mask.GetMaskedConnectionCount(outputIndex: 0));
        Assert.AreEqual(0, mask.GetUnmaskedConnectionCount(outputIndex: 0));
        Assert.IsTrue(mask.GetConnectionMaskVector(outputIndex: 0).All(isMasked => isMasked));
        Assert.AreEqual(0, mask.GetMaskedConnectionCount(outputIndex: 1));
    }

    [TestMethod]
    public void UnmaskingAConnectionRestoresIt()
    {
        ConnectionMaskBuilder builder = new(inputCount: 3, outputCount: 2);

        builder.MaskAllConnections(neuronIndex: 0);
        builder.SetIsConnectionMasked(neuronIndex: 0, connectionIndex: 1, isMasked: false);

        IConnectionMask mask = builder.Build();

        Assert.IsFalse(mask.IsConnectionMasked(inputIndex: 1, outputIndex: 0));
        Assert.AreEqual(2, mask.GetMaskedConnectionCount(outputIndex: 0));
        Assert.AreEqual(1, mask.GetUnmaskedConnectionCount(outputIndex: 0));
    }

    [TestMethod]
    public void AMaskIsUnmovedByLaterChangesToTheBuilderThatMadeIt()
    {
        ConnectionMaskBuilder builder = new(inputCount: 3, outputCount: 2);

        builder.SetIsConnectionMasked(neuronIndex: 0, connectionIndex: 0, isMasked: true);

        IConnectionMask mask = builder.Build();

        builder.SetIsConnectionMasked(neuronIndex: 0, connectionIndex: 1, isMasked: true);

        Assert.IsFalse(mask.IsConnectionMasked(inputIndex: 1, outputIndex: 0));
        Assert.AreEqual(1, mask.GetMaskedConnectionCount(outputIndex: 0));
    }

    [TestMethod]
    public void AnIndexOutsideTheMaskIsRefused()
    {
        IConnectionMask mask = new ConnectionMaskBuilder(inputCount: 3, outputCount: 2).Build();

        Asserter.AssertThat(() => mask.IsConnectionMasked(inputIndex: 3, outputIndex: 0))
            .Throws<ArgumentOutOfRangeException>();

        Asserter.AssertThat(() => mask.IsConnectionMasked(inputIndex: 0, outputIndex: 2))
            .Throws<ArgumentOutOfRangeException>();

        Asserter.AssertThat(() => mask.GetMaskedConnectionCount(outputIndex: -1))
            .Throws<ArgumentOutOfRangeException>();
    }

    #endregion
}
