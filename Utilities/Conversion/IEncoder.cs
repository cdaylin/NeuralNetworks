using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Daylin.Utilities.Conversion;

/// <summary>
/// Responsible for converting (i.e.: encoding) objects from one type to another.
/// </summary>
/// <typeparam name="TSource">
/// Type to be encoded.
/// </typeparam>
/// <typeparam name="TEncoded">
/// Encoded type.
/// </typeparam>
public interface IEncoder<TSource, TEncoded>
{
    TEncoded Encode(TSource sourceObject);
}