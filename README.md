# NeuralNetworks

A feed-forward neural network library written in C# from first principles -- no machine learning
framework underneath it. Layers, activation functions, weight initialization, and backpropagation are
all implemented directly.

I wrote this to keep my understanding of the fundamentals concrete rather than to compete with
PyTorch or TensorFlow. If you need to train a production model, use one of those. If you want to read
how backpropagation actually works, this is a few thousand lines you can follow end to end.

## What is here

The library is organized around a small number of composable pieces.

**Layers.** `DenseLayer` is the fully-connected case. `SparseLayer` takes a connection mask, so a
layer can be wired arbitrarily rather than completely. `ConvolutionalLayer` shares one weight vector
across a moving span. Above those sit composite layers -- `SequenceLayer` chains layers,
`ConcatenationLayer` runs them side by side and joins the outputs, and `ConnectionRoutingLayer`
permutes values between layers through an `IndexMap`.

**Layer templates.** A network is described by templates and then built, so the description of a
network is separable from the network itself. `NeuralNetworkBuilder` is the usual entry point.

**Activation functions.** Sigmoid, hyperbolic tangent, softsign, ReLU, and EANAF, each able to return
its value, its derivative, or both from one call. Training needs both, and for several of them the
derivative falls out of the value -- the logistic sigmoid's is its own output times one minus that
output -- so the combined call costs one `exp` where two separate calls cost two.

**Weight initialization.** Glorot and He, both uniform, each capped at a maximum range because for
neurons with a small fan-in the raw formula produces an excessively wide interval.

**Backpropagation.** A mean-squared-error error function, a learning-rate schedule behind an interface,
and a training pass that computes each layer's contribution on the way back.

## Using it

A network is described layer by layer, built, and then trained one sample at a time. This trains
exclusive-or, the smallest problem a network without a hidden layer cannot solve:

```csharp
using Daylin.NeuralNetworks;
using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.Backpropagation;
using Daylin.NeuralNetworks.WeightInitialization;

NeuralNetworkBuilder builder = new(inputCount: 2);

builder.AddDenseLayer(
    outputCount: 8,
    activationFunction: ActivationFunctions.Tanh,
    weightInitializer: WeightInitializers.Glorot);

builder.AddDenseLayer(
    outputCount: 1,
    activationFunction: ActivationFunctions.Tanh,
    weightInitializer: WeightInitializers.Glorot);

INeuralNetwork network = builder.BuildNetwork();

BackpropagationSettings settings = new(
    learningRate: 0.1f,
    errorFunction: ErrorFunctions.MeanSquaredError);

float[][] inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
float[][] targets = [[0], [1], [1], [0]];

for (int epoch = 0; epoch < 2000; epoch++)
    for (int index = 0; index < inputs.Length; index++)
        await network.TrainAsync(settings, inputs[index], targets[index]);

foreach (float[] input in inputs)
{
    IEnumerable<float> output = await network.CalculateOutputAsync(input);

    Console.WriteLine($"{input[0]} xor {input[1]} -> {output.First():F2}");
}
```

which prints roughly:

```
0 xor 0 -> 0.00
0 xor 1 -> 0.98
1 xor 0 -> 0.98
1 xor 1 -> 0.00
```

Only roughly, because the starting weights are random. The weight initializer is paired with the
activation function deliberately: Glorot for the symmetric functions, He for ReLU and EANAF.

## Building and running

Requires the .NET 9 SDK or later.

```bash
dotnet build
```

```bash
dotnet test
```

The test suite trains 24 networks to convergence -- every activation function against identity, NOT,
AND, and exclusive-or, plus four deeper architectures -- and asserts that each one learns the function
within a bounded number of epochs. It takes a few seconds.

Because every network starts from random weights, the suite is deliberately nondeterministic, and the
tests are marked as such. The layer sizes are chosen to make a failure rare rather than impossible;
the next section is about how rare.

## Why the hidden layers are sized as they are

Backpropagation is gradient descent, so whether a particular network escapes a local optimum depends
on the weights it started from. That makes a single passing run weak evidence: a configuration can be
robust, or it can merely have been lucky, and one run cannot tell you which.

The layer sizes here were chosen by measuring the difference. Training the same configuration from
many different starting points and counting how often it converges gives a rate rather than a verdict.
A ReLU hidden layer taking the raw inputs converged 48% of the time at 4 neurons, 88% at 8, and always
at 12 or more -- so the tests use 12. Behind a softsign layer the same layer is more forgiving: 80% at
4 neurons, 98% at 8, always at 10.

Those numbers are recorded beside the constants they justify, so a future reader knows what to
re-measure rather than what to trust.

## Layout

| Project | |
| --- | --- |
| `NeuralNetworks` | The library. This is the part worth reading. |
| `NeuralNetworksTests` | MSTest suite; trains real networks rather than mocking anything. |
| `Utilities`, `TestUtilities` | Shared general-purpose libraries, included so the solution builds standalone; not what this repository is meant to show. |

## On authorship

I wrote this library myself, by hand. It is not generated code, and that is rather the point of
it. It exists as a record of understanding the mathematics well enough to implement it.

This library was finished before it entered the private repository it is published from, arriving
there whole in a February 2025 commit that consolidated several older repositories into one. Since
that commit I have changed it by roughly a hundred lines, some of that with the assistance of Claude
Code. The architecture and the mathematics are entirely mine.

## About this repository

This repository is generated from a private one, which is where the code is actually edited. Each
commit here is a publication rather than a step in the library's development, so the history is a
record of releases and not of the work that produced it. Pull requests cannot be merged back; if you
find something wrong, an issue is the useful thing to open.

## License

MIT. See [LICENSE](LICENSE).
