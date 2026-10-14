using Daylin.NeuralNetworks.Backpropagation;
using Daylin.NeuralNetworks.Backpropagation.Optimization;

using System.Text.Json;

namespace Daylin.NeuralNetworks.Serialization;

/// <summary>
/// Tests what the library does with a saved document that disagrees with itself.
/// </summary>
/// <remarks>
/// A layer holds its weights in arrays sized from its input and output counts, and a document carries both, so
/// each layer checks the two against each other as it is read.  What a document that disagrees would otherwise
/// build depends on which way it disagrees:  a layer that answers and then fails partway through a training
/// batch, or one that silently never connects an input value.  The session's own refusals are here too -- a
/// document saved before the cost function was renamed, and optimizer state no session could have written --
/// since they are the same question asked of the document that holds the layers.
/// </remarks>
[TestClass]
public sealed class LayerDocumentTests
{
    #region Public

    [TestMethod]
    public void ADenseLayerRefusesAWeightRowThatDoesNotSpanItsInputs() =>
        AssertDocumentIsRefused<ArgumentOutOfRangeException>(
            DenseDocument(biasCount: 2, weightRows: 2, weightsPerRow: 3));

    [TestMethod]
    public void ADenseLayerRefusesFewerWeightRowsThanOutputs() =>
        AssertDocumentIsRefused<ArgumentOutOfRangeException>(
            DenseDocument(biasCount: 2, weightRows: 1, weightsPerRow: 2));

    [TestMethod]
    public void ADenseLayerRefusesABiasForEveryOtherOutput() =>
        AssertDocumentIsRefused<ArgumentOutOfRangeException>(
            DenseDocument(biasCount: 1, weightRows: 2, weightsPerRow: 2));

    [TestMethod]
    public void ADenseLayerReadsADocumentWhoseArraysFit()
    {
        INeuralNetwork network = NetworkDocument
            .FromJson(DenseDocument(biasCount: 2, weightRows: 2, weightsPerRow: 2))
            .BuildNetwork();

        Assert.AreEqual(2, network.OutputCount, "The layer's output count.");
    }

    [TestMethod]
    public void AConvolutionalLayerRefusesAWeightVectorThatDoesNotSpanItsInputs()
    {
        // Four inputs over two output values is a span of two, so three weights describe no such layer.
        AssertDocumentIsRefused<ArgumentOutOfRangeException>(
            ConvolutionalDocument(inputCount: 4, outputCount: 2, weightCount: 3));
    }

    [TestMethod]
    public void AConvolutionalLayerRefusesInputsThatDoNotDivideAmongItsOutputs()
    {
        // Five inputs over two output values leaves a remainder, which both constructors refuse through the one
        //  guard they share -- an ArgumentException naming the input count, rather than a range refusal, since
        //  no count here is out of range against another.
        AssertDocumentIsRefused<ArgumentException>(
            ConvolutionalDocument(inputCount: 5, outputCount: 2, weightCount: 2));
    }

    [TestMethod]
    public void ASparseLayerRefusesAMaskThatDescribesAnotherLayersShape()
    {
        // The mask says three input values and the layer says two.
        AssertDocumentIsRefused<ArgumentOutOfRangeException>(SparseDocument(inputCount: 2, maskInputCount: 3));
    }

    [TestMethod]
    public async Task ASparseLayerReadsADocumentWhoseRowsMatchItsMaskAsync()
    {
        INeuralNetwork network = NetworkDocument
            .FromJson(SparseDocument(inputCount: 2, maskInputCount: 2))
            .BuildNetwork();

        // The second input value is masked away from the one output value, so only the first reaches it.
        await NetworkAssert.OutputIsAsync(network, [1f, 1f], Math.Tanh((0.5 * 1) + 0.1));
    }

    [TestMethod]
    public void ASessionSavedBeforeTheCostFunctionIsRefusedByName()
    {
        // The library once named this setting a cost derivative.  A document carrying that name would leave the
        //  settings without a cost at all, and the failure would name a constructor parameter no caller supplied.
        string session = $$"""
            {
              "Network": {{DenseDocument(biasCount: 2, weightRows: 2, weightsPerRow: 2)}},
              "Settings": {
                "CostDerivative": { "$type": "MeanSquaredError" },
                "Optimizer": { "$type": "GradientDescent" },
                "LearningRateScheduler": { "$type": "Constant", "LearningRate": 0.5 },
                "MaxGradientNorm": null
              },
              "CompletedEpochCount": 3
            }
            """;

        ArgumentException exception = Assert.ThrowsExactly<ArgumentException>(() => TrainingSession.Resume(session));

        StringAssert.Contains(exception.Message, "CostDerivative", "The message names the setting.");

        StringAssert.Contains(
            exception.Message,
            nameof(TrainingSettings.CostFunction),
            "The message names what is read now.");
    }

    [TestMethod]
    public void ASparseLayerRefusesAMaskDescribingAnotherNumberOfOutputValues()
    {
        // The input half of this was covered and the output half was not:  a mask of two output values against a
        //  layer of one describes a different layer, and its extra row would never be read.
        AssertDocumentIsRefused<ArgumentOutOfRangeException>(
            SparseDocument(inputCount: 2, maskInputCount: 2, maskOutputCount: 2));
    }

    [TestMethod]
    public void ASparseLayerRefusesAMaskRowThatDoesNotSpanItsInputs()
    {
        // The counts all agree and the row does not:  a row of one value against two input values builds a mask
        //  that answers for the connection it has and throws on the one it does not, part way through the first
        //  forward pass, and makes the unmasked count disagree with the weight row the layer holds.
        AssertDocumentIsRefused<ArgumentOutOfRangeException>(
            SparseDocument(inputCount: 2, maskInputCount: 2, maskRowLength: 1));
    }

    [TestMethod]
    public void AnAdamStateIsRefusedWhereItsStepCountOrItsSecondMomentsCouldNotHaveBeenWritten()
    {
        // Both reach the square root of the corrected second moment, so both turn every parameter of the group
        //  into a value that is not a number at the first step, and nothing downstream looks.
        AssertSessionStateIsRefused(
            """{ "$type": "Adam", "FirstMoments": [0.01], "SecondMoments": [0.0009], "StepCount": -1 }""",
            "a negative step count");

        AssertSessionStateIsRefused(
            """{ "$type": "Adam", "FirstMoments": [0.01], "SecondMoments": [-0.0009], "StepCount": 2 }""",
            "a negative second moment");
    }

    [TestMethod]
    public async Task TheNetworkOfASessionThatCannotBeResumedIsStillReadableAsync()
    {
        // The refusal above tells a caller the weights are not lost, and this is the route it names.  A document
        //  the library can no longer resume still carries a network whose own document never changed.
        string session = $$"""
            {
              "Network": {{DenseDocument(biasCount: 2, weightRows: 2, weightsPerRow: 2)}},
              "Settings": {
                "CostDerivative": { "$type": "MeanSquaredError" },
                "Optimizer": { "$type": "GradientDescent" },
                "LearningRateScheduler": { "$type": "Constant", "LearningRate": 0.5 },
                "MaxGradientNorm": null
              },
              "CompletedEpochCount": 3
            }
            """;

        INeuralNetwork network = TrainingSession.ReadNetwork(session);

        await NetworkAssert.OutputIsAsync(
            network,
            [1f, 1f],
            Math.Tanh((0.5 * 1) + (0.5 * 1) + 0.1),
            Math.Tanh((0.5 * 1) + (0.5 * 1) + 0.1));

        // And it can be trained, which is the point of getting it back.
        TrainingSession started = TrainingSession.Adopt(
            network,
            TrainingFixture.CreateSettings(Optimizers.GradientDescent));

        Assert.AreEqual(0, started.CompletedEpochCount, "A session started from it has completed no epochs.");
    }

    [TestMethod]
    public void ASessionSavedWithACostFunctionResumes()
    {
        TrainingSession session = TrainingSession.Start(
            TrainingFixture.BuildNeuronNetwork(),
            TrainingFixture.CreateSettings(Optimizers.GradientDescent));

        TrainingSession resumed = TrainingSession.Resume(session.ToJson());

        Assert.AreEqual(
            session.Settings.CostFunction.ShortName,
            resumed.Settings.CostFunction.ShortName,
            "The resumed session's cost function.");
    }

    #endregion

    #region Private Static

    /// <summary>
    /// Asserts that a document is refused, by the exact exception the layer refuses it with.
    /// </summary>
    /// <remarks>
    /// Exact rather than assignable, because the type is what distinguishes a count that disagrees with another
    /// count from a value the layer rules out whatever it is measured against.
    /// </remarks>
    /// <typeparam name="TException">
    /// The exception the layer is expected to throw.
    /// </typeparam>
    /// <param name="document">
    /// The document to read.
    /// </param>
    private static void AssertDocumentIsRefused<TException>(string document)
        where TException : Exception
    {
        Assert.ThrowsExactly<TException>(
            () => NetworkDocument.FromJson(document).BuildNetwork(),
            "The document was read rather than refused.");
    }

    private static string DenseDocument(int biasCount, int weightRows, int weightsPerRow)
    {
        string biases = string.Join(", ", Enumerable.Repeat("0.1", biasCount));

        string rows = string.Join(
            ", ",
            Enumerable.Repeat($"[{string.Join(", ", Enumerable.Repeat("0.5", weightsPerRow))}]", weightRows));

        return $$"""
            {
              "$type": "NeuralNetwork",
              "NetworkLayer": {
                "$type": "DenseLayer",
                "BiasVector": [{{biases}}],
                "WeightMatrix": [{{rows}}],
                "InputCount": 2,
                "OutputCount": 2,
                "ActivationFunction": { "$type": "Tanh" },
                "WeightInitializer": { "$type": "GlorotUniform", "MaxRange": 1 }
              }
            }
            """;
    }

    private static string ConvolutionalDocument(int inputCount, int outputCount, int weightCount)
    {
        string weights = string.Join(", ", Enumerable.Repeat("0.5", weightCount));

        return $$"""
            {
              "$type": "NeuralNetwork",
              "NetworkLayer": {
                "$type": "ConvolutionalLayer",
                "WeightVector": [{{weights}}],
                "Bias": 0.1,
                "InputCount": {{inputCount}},
                "OutputCount": {{outputCount}},
                "ActivationFunction": { "$type": "Tanh" },
                "WeightInitializer": { "$type": "GlorotUniform", "MaxRange": 1 }
              }
            }
            """;
    }

    /// <summary>
    /// Builds a network document holding one sparse layer of a single output value.
    /// </summary>
    /// <param name="inputCount">
    /// Number of input values the layer declares.
    /// </param>
    /// <param name="maskInputCount">
    /// Number of input values the mask declares.
    /// </param>
    /// <param name="maskOutputCount">
    /// Number of output values the mask declares, against the layer's one.
    /// </param>
    /// <param name="maskRowLength">
    /// Number of values in each mask row, or <see langword="null"/> for a row spanning the mask's input count.
    /// </param>
    /// <returns>
    /// Returns the document.
    /// </returns>
    private static string SparseDocument(
        int inputCount,
        int maskInputCount,
        int maskOutputCount = 1,
        int? maskRowLength = null)
    {
        string maskRow = string.Join(
            ", ",
            Enumerable
                .Range(0, maskRowLength ?? maskInputCount)
                .Select(index => index == 0 ? "false" : "true"));

        string maskRows = string.Join(", ", Enumerable.Repeat($"[{maskRow}]", maskOutputCount));

        return $$"""
            {
              "$type": "NeuralNetwork",
              "NetworkLayer": {
                "$type": "SparseLayer",
                "WeightMatrix": [[0.5]],
                "BiasVector": [0.1],
                "ConnectionMask": {
                  "$type": "ConnectionMask",
                  "InputCount": {{maskInputCount}},
                  "OutputCount": {{maskOutputCount}},
                  "MaskMatrix": [{{maskRows}}]
                },
                "InputCount": {{inputCount}},
                "OutputCount": 1,
                "ActivationFunction": { "$type": "Tanh" },
                "WeightInitializer": { "$type": "GlorotUniform", "MaxRange": 1 }
              }
            }
            """;
    }

    /// <summary>
    /// Asserts that a session document carrying given optimizer state is refused.
    /// </summary>
    /// <param name="state">
    /// The state for the layer's one bias group, as JSON.
    /// </param>
    /// <param name="fault">
    /// What is wrong with it, for the assertion's message.
    /// </param>
    private static void AssertSessionStateIsRefused(string state, string fault)
    {
        string session = $$"""
            {
              "Network": {{SingleNeuronDocument()}},
              "Settings": {
                "CostFunction": { "$type": "MeanSquaredError" },
                "Optimizer": { "$type": "Adam", "FirstMomentDecay": 0.9, "SecondMomentDecay": 0.999,
                  "Epsilon": 1e-8 },
                "LearningRateScheduler": { "$type": "Constant", "LearningRate": 0.5 },
                "MaxGradientNorm": null
              },
              "CompletedEpochCount": 1,
              "OptimizerState": [{{state}}, {{state}}]
            }
            """;

        Assert.ThrowsExactly<ArgumentOutOfRangeException>(
            () => TrainingSession.Resume(session),
            $"A document carrying {fault} was resumed.");
    }

    /// <summary>
    /// Builds a network document holding one dense layer of a single neuron over one input value.
    /// </summary>
    /// <returns>
    /// Returns the document, whose layer holds a bias group and one weight group.
    /// </returns>
    private static string SingleNeuronDocument() =>
        """
        {
          "$type": "NeuralNetwork",
          "NetworkLayer": {
            "$type": "DenseLayer",
            "BiasVector": [0.1],
            "WeightMatrix": [[0.5]],
            "InputCount": 1,
            "OutputCount": 1,
            "ActivationFunction": { "$type": "Tanh" },
            "WeightInitializer": { "$type": "GlorotUniform", "MaxRange": 1 }
          }
        }
        """;

    #endregion
}
