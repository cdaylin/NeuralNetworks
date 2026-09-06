namespace Daylin.NeuralNetworks.Backpropagation;

public class BackpropagationSettings
{
    public BackpropagationSettings(
        float learningRate,
        ErrorFunction errorFunction)
    {
        learningRate.ThrowIfNotPositive();
        errorFunction.ThrowIfNull();

        LearningRate = learningRate;
        ErrorFunction = errorFunction;
    }

    public float LearningRate { get; }

    public ErrorFunction ErrorFunction { get; }
}
