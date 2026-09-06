using System.Text.Json.Serialization;

namespace Daylin.NeuralNetworks;

[JsonPolymorphic]
[JsonDerivedType(typeof(ConnectionMask), typeDiscriminator: nameof(ConnectionMask))]
public interface IConnectionMask
{
    int OutputCount { get; }

    int InputCount { get; }

    int GetMaskedConnectionCount(int outputIndex);

    int GetUnmaskedConnectionCount(int outputIndex);

    bool IsConnectionMasked(int inputIndex, int outputIndex);

    IReadOnlyList<bool> GetConnectionMaskVector(int outputIndex);
}
