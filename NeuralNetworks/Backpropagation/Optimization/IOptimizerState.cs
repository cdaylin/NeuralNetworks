using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Backpropagation.Optimization;

/// <summary>
/// The state an optimizer carries from one training step to the next, for one group of parameters.
/// </summary>
/// <remarks>
/// A group of parameters is a layer's bias vector, or the connection weights of one of its output values.  An
/// optimizer that needs no state creates none.
/// <para>
/// A layer holds the state, because only a layer knows the shape of its own parameters;  the training session
/// owns it, and is what serializes it.  A network saved on its own therefore carries none of it:  load a network
/// for inference and that is all there is, and resume a session to continue training where it left off.
/// </para>
/// </remarks>
[JsonPolymorphic]
[JsonDerivedType(typeof(MomentumOptimizerState), typeDiscriminator: MomentumOptimizer.Discriminator)]
[JsonDerivedType(typeof(AdamOptimizerState), typeDiscriminator: AdamOptimizer.Discriminator)]
public interface IOptimizerState
{
    /// <summary>
    /// Number of parameters the state is held for.
    /// </summary>
    int ParameterCount { get; }
}
