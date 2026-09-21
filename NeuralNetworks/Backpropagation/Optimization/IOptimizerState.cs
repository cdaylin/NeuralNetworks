using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks.Backpropagation.Optimization;

/// <summary>
/// The state an optimizer carries from one training step to the next, for one group of parameters.
/// </summary>
/// <remarks>
/// A group of parameters is a layer's bias vector, or the connection weights of one of its output values.  An
/// optimizer that needs no state creates none.
/// <para>
/// State belongs to the layer holding the parameters and is serialized with them, so that a network saved
/// partway through training continues where it left off rather than beginning its optimizer afresh.
/// </para>
/// </remarks>
[JsonPolymorphic]
[JsonDerivedType(typeof(MomentumOptimizerState), typeDiscriminator: "Momentum")]
[JsonDerivedType(typeof(AdamOptimizerState), typeDiscriminator: "Adam")]
public interface IOptimizerState
{
    /// <summary>
    /// Number of parameters the state is held for.
    /// </summary>
    int ParameterCount { get; }
}
