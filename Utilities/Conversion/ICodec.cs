namespace Daylin.Utilities.Conversion;

/// <summary>
/// Responsible for converting objects between two types (i.e.: encoding and decoding).
/// </summary>
/// <typeparam name="TSource">
/// Type to be encoded.
/// </typeparam>
/// <typeparam name="TEncoded">
/// Encoded type.
/// </typeparam>
public interface ICodec<TSource, TEncoded> : IEncoder<TSource, TEncoded>
{
    TSource Decode(TEncoded encodedObject);
}
