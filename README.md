# NeuralNetworks

A feed-forward neural network library written in C# from first principles -- no machine learning
framework underneath it.  Layers, activation functions, weight initialization, and backpropagation are
all implemented directly.

I wrote this to keep my understanding of the fundamentals concrete rather than to compete with
PyTorch or TensorFlow.  If you need to train a production model, use one of those.  If you want to read
how backpropagation actually works, this is a few thousand lines you can follow end to end.

## What is here

The library is organized around a small number of composable pieces.

**Layers.**  `DenseLayer` is the fully-connected case.  `SparseLayer` takes a connection mask, so a
layer can be wired arbitrarily rather than completely.  `ConvolutionalLayer` shares one weight vector
across a moving span.  Above those sit composite layers -- `SequenceLayer` chains layers,
`ConcatenationLayer` runs them side by side and joins the outputs, and `ConnectionRoutingLayer`
permutes values between layers through an `IndexMap`.

**Layer templates.**  A network is described by templates and then built, so the description of a
network is separable from the network itself.  `NeuralNetworkBuilder` is the usual entry point.

**Activation functions.**  Sigmoid, hyperbolic tangent, softsign, ReLU, and EANAF, each able to return
its value, its derivative, or both from one call.  Training needs both, and for several of them the
derivative falls out of the value -- the logistic sigmoid's is its own output times one minus that
output -- so the combined call costs one `exp` where two separate calls cost two.

**Weight initialization.**  Glorot and He, both uniform, each capped at a maximum range because for
neurons with a small fan-in the raw formula produces an excessively wide interval.

**Training.**  Backpropagation gathers each parameter's gradient on the way back, and a
`TrainingSession` turns those gradients into steps:  one sample at a time, a batch averaged over its
samples, or an epoch taken in batches.  Plain gradient descent, momentum, and Adam sit behind one
optimizer interface, with a learning rate that can decay from epoch to epoch and an optional limit on
the norm of a batch's gradients.  The session holds the optimizer's state and the epoch count, and
saves them with the network so training can resume where it stopped; a network saved on its own
carries nothing about training.  Training is reproducible to the last bit:  the same starting weights,
settings, and samples leave the same weights every time.

## Using it

A network is described layer by layer, built, and then trained through a session.  This trains
exclusive-or, the smallest problem a network without a hidden layer cannot solve:

```csharp
using Daylin.NeuralNetworks;
using Daylin.NeuralNetworks.Activation;
using Daylin.NeuralNetworks.Backpropagation;
using Daylin.NeuralNetworks.Backpropagation.LearningRate;
using Daylin.NeuralNetworks.Backpropagation.Optimization;
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

TrainingSettings settings = new(
    costDerivative: CostDerivatives.MeanSquaredError,
    optimizer: Optimizers.Adam(),
    learningRateScheduler: LearningRateSchedulers.ConstantLearningRateScheduler(0.05f));

TrainingSession session = TrainingSession.Adopt(network, settings);

float[][] inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
float[][] targets = [[0], [1], [1], [0]];

for (int epoch = 0; epoch < 200; epoch++)
    await session.TrainEpochAsync(inputs, targets, batchSize: 4);

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

Only roughly, because the starting weights are random.  The weight initializer is paired with the
activation function deliberately:  Glorot for the symmetric functions, He for ReLU and EANAF.

`Adopt` trains the network it is given, which is why the example can read its outputs afterwards.
`Start` trains a copy instead and leaves the caller's network alone; the network that learns is then
`session.Network`.  `TrainSampleAsync` and `TrainBatchAsync` take one sample or one batch rather than an
epoch.  `session.ToJson()` saves the network together with the optimizer's state and the epoch count,
and `TrainingSession.Resume` picks the training up from that document.

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
within a bounded number of epochs.  Its other tests state exact results rather than convergence:  the
weights one training step leaves in each layer type, each optimizer's formula, how a batch averages,
the learning-rate schedules, gradient clipping, what a saved network and a saved session carry, and
that training the same network on the same samples twice leaves the same bits.  All 106 run in a few
seconds.

Every network is built from a **seeded** weight initializer, so each run trains exactly the same
networks and a failure is a real signal rather than an unlucky draw.  The seeds are arbitrary and
meant to stay that way:  one chosen because it passes would prove only that it still passes, and would
hide a regression that broke most other seeds.  When a seed fails, the configuration is what should
change — which is how the ReLU layer came to be sized as it is.

## Why the hidden layers are sized as they are

Backpropagation is gradient descent, so whether a particular network escapes a local optimum depends
on the weights it started from.  That makes a single passing run weak evidence:  a configuration can be
robust, or it can merely have been lucky, and one run cannot tell you which.

The layer sizes here were chosen by measuring the difference.  Training the same configuration from
many different starting points and counting how often it converges gives a rate rather than a verdict.
A ReLU hidden layer taking the raw inputs converged 48% of the time at 4 neurons, 88% at 8, and always
at 12 or more -- so the tests use 12.  Behind a softsign layer the same layer is more forgiving:  80% at
4 neurons, 98% at 8, always at 10.

Those numbers are recorded beside the constants they justify, so a future reader knows what to
re-measure rather than what to trust.

## Layout

| Project | |
| --- | --- |
| `NeuralNetworks` | The library.  This is the part worth reading. |
| `NeuralNetworksTests` | MSTest suite; trains real networks rather than mocking anything. |
| `Utilities`, `TestUtilities` | Shared general-purpose libraries, included so the solution builds standalone; not what this repository is meant to show. |

## On authorship

I wrote the original library myself, by hand:  the layers, the activation functions, weight
initialization, and backpropagation.  That part is not generated code, and that is rather the point of
it.  It exists as a record of understanding the mathematics well enough to implement it.

The original library was finished before it entered the private repository it is published from,
arriving there whole in a February 2025 commit that consolidated several older repositories into one.
At that commit it was about 3,300 lines of C#, with about 350 lines of tests.

Since then it has grown substantially, and most of that growth was written by Claude Code at my
direction.  Batch training, the three optimizers, the learning-rate schedules, the training session
and what it saves, gradient clipping, and nearly all of the tests came that way, most of it in
September 2026.  The library is now about 4,900 lines and its tests about 3,500.  I set the
requirements, made the design decisions, and reviewed the work; the code for those additions is
largely Claude's.

## About this repository

This repository is generated from a private one, which is where the code is actually edited.  Each
commit here is a publication rather than a step in the library's development, so the history is a
record of releases and not of the work that produced it.  Pull requests cannot be merged back; if you
find something wrong, an issue is the useful thing to open.

## License

MIT.  See [LICENSE](LICENSE).
