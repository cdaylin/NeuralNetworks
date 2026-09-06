namespace Daylin.NeuralNetworks.Backpropagation;

/// <summary>
/// Input/output pairs for simple logic functions.
/// </summary>
public readonly struct BackpropagationTestData
{
    #region Static Construction

    static BackpropagationTestData()
    {
        float[][] input;
        float[][] target;

        // identity
        input = new float[2][];
        input[0] = [0];
        input[1] = [1];

        target = input;

        Identity = new BackpropagationTestData()
        {
            ShortName = "Identity",
            InputData = input,
            TargetData = target
        };

        // not
        input = new float[2][];
        input[0] = [1];
        input[1] = [0];

        target = new float[2][];
        target[0] = [0];
        target[1] = [1];

        Not = new BackpropagationTestData()
        {
            ShortName = "Not",
            InputData = input,
            TargetData = target
        };

        // and
        input = new float[4][];

        input[0] = [1, 1];
        input[1] = [1, 0];
        input[2] = [0, 1];
        input[3] = [0, 0];

        target = new float[4][];

        target[0] = [1];
        target[1] = [0];
        target[2] = [0];
        target[3] = [0];

        And = new BackpropagationTestData()
        {
            ShortName = "And",
            InputData = input,
            TargetData = target
        };

        // exclusive-or
        input = new float[4][];

        input[0] = [1, 1];
        input[1] = [1, 0];
        input[2] = [0, 1];
        input[3] = [0, 0];

        target = new float[4][];

        target[0] = [0];
        target[1] = [1];
        target[2] = [1];
        target[3] = [0];

        XOr = new BackpropagationTestData()
        {
            ShortName = "XOr",
            InputData = input,
            TargetData = target
        };
    }

    #endregion

    #region Public Static

    public static BackpropagationTestData Identity { get; }

    public static BackpropagationTestData Not { get; }

    public static BackpropagationTestData And { get; }

    public static BackpropagationTestData XOr { get; }

    #endregion

    #region Construction

    public BackpropagationTestData()
    {
        InputData = Array.Empty<IReadOnlyList<float>>();
        TargetData = Array.Empty<IReadOnlyList<float>>();
    }

    #endregion

    #region Public

    public required string ShortName { get; init; }

    public required IReadOnlyList<IReadOnlyList<float>> InputData { get; init; }

    public required IReadOnlyList<IReadOnlyList<float>> TargetData { get; init; }

    public int InputCount => InputData.First().Count;

    public int OutputCount => TargetData.First().Count;

    #endregion

}
