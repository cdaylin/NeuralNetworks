namespace Daylin.Utilities.Primitives;

/// <summary>
/// Distinguishes between "one" vs. "many".
/// </summary>
/// <remarks>
/// This usage of "cardinality" corresponds closely to the usage of "cardinality" in database table relationships
/// (i.e.: "0 to 1" vs. "0 to many".  Depending upon context, this usage of "cardinality" can  also be used to 
/// distinguish between "1" vs "many".
/// <para>
/// Note that the word "multiplicity" is sometimes used as a synonym for "cardinality", although "cardinality" as
/// used in database table relationships and "multiplicity" as used in UML do differ meaning.
/// </para>
/// </remarks>
public enum Cardinality
{
    /// <summary>
    /// One.
    /// </summary>
    Single,

    /// <summary>
    /// One or more.
    /// </summary>
    Multiple
}

public static class CardinalityExtensions
{
    public static bool IsSingle(this Cardinality cardinality)
    {
        return cardinality is Cardinality.Single;
    }

    public static bool IsMultiple(this Cardinality cardinality)
    {
        return cardinality is Cardinality.Multiple;
    }
}
