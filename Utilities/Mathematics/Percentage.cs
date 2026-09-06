namespace Daylin.Utilities.Mathematics;

/// <summary>
/// Represents a non-negative percentage.
/// </summary>
public readonly struct Percentage : IComparable<Percentage>, IEquatable<Percentage>
{
    public static readonly Percentage Zero = new(0M);
    public static readonly Percentage Ten = new(10M);
    public static readonly Percentage Fifty = new(50M);
    public static readonly Percentage OneHundred = new(100M);

    public static Percentage operator *(Percentage left, Percentage right)
    {
        return new Percentage(left.DecimalValue * right.DecimalValue);
    }

    public static Percentage operator +(Percentage left, Percentage right)
    {
        return new Percentage(left.DecimalValue + right.DecimalValue);
    }

    public static bool operator ==(Percentage left, Percentage right)
    {
        return Equals(left, right);
    }

    public static bool operator !=(Percentage left, Percentage right)
    {
        return !(left == right);
    }

    public Percentage() : this(0M)
    {
    }

    public Percentage(decimal value)
    {
        value.ThrowIfNegative();

        DecimalValue = value;
    }

    public decimal DecimalValue { get; }

    public int CompareTo(Percentage other)
    {
        return DecimalValue.CompareTo(other.DecimalValue);
    }

    public bool Equals(Percentage other)
    {
        return DecimalValue.Equals(other.DecimalValue);
    }

    public override bool Equals(object? other)
    {
        return other is Percentage otherPercentage && Equals(otherPercentage);
    }

    public override int GetHashCode()
    {
        return DecimalValue.GetHashCode();
    }

    public override string ToString()
    {
        return $"{DecimalValue}%";
    }
}
