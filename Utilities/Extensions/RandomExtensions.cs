using System.Numerics;

namespace Daylin.Utilities.Extensions;

/// <summary>
/// Extension methods for type <see cref="Random"/>.
/// </summary>
public static class RandomExtensions
{
    public static TItem SelectWeighted<TItem, TWeight>(
        this Random randomNumberGenerator,
        IReadOnlyList<(TItem item, TWeight weight)> items)
        where TWeight : INumber<TWeight>
    {
        randomNumberGenerator.ThrowIfNull();
        items.ThrowIfNull();
        items.ThrowIfEmpty();

        foreach ((TItem item, TWeight weight) in items)
            weight.ThrowIfNotPositive();

        TWeight weightSum = items.Select(tuple => tuple.weight).Sum();

        // get random number less than sum of weights
        TWeight randomValue = TWeight.CreateSaturating(
            randomNumberGenerator.NextDouble() * double.CreateChecked(weightSum));

        // find item corresponding to random number
        TWeight cumulativeSum = TWeight.Zero;

        for (int index = 0; index < items.Count - 1; index++)
        {
            cumulativeSum += items[index].weight;

            if (cumulativeSum > randomValue)
                return items[index].item;
        }

        return items[items.Count - 1].item;
    }

    public static TItem Select<TItem>(
        this Random randomNumberGenerator,
        IReadOnlyList<TItem> items)
    {
        randomNumberGenerator.ThrowIfNull();
        items.ThrowIfNull();
        items.ThrowIfEmpty();

        return items[randomNumberGenerator.Next(items.Count)];
    }

    public static T NextPositiveInteger<T>(this Random randomNumberGenerator, T maxValue)
        where T : IBinaryInteger<T>
    {
        randomNumberGenerator.ThrowIfNull();
        maxValue.ThrowIfNotPositive();

        double randomDouble = randomNumberGenerator.NextDouble() * double.CreateChecked(maxValue);

        return T.CreateChecked(randomDouble) + T.One;
    }

    public static T NextPositiveNumber<T>(this Random randomNumberGenerator, T maxValue)
        where T : IFloatingPointIeee754<T>
    {
        randomNumberGenerator.ThrowIfNull();
        maxValue.ThrowIfNotPositive();

        double randomDouble = randomNumberGenerator.NextDouble() * double.CreateChecked(maxValue);

        return T.Min(maxValue, T.CreateChecked(randomDouble) + T.Epsilon);
    }

    public static decimal NextPositiveDecimal(this Random randomNumberGenerator, decimal maxValue)
    {
        randomNumberGenerator.ThrowIfNull();
        maxValue.ThrowIfNotPositive();

        double randomDouble = randomNumberGenerator.NextDouble() * double.CreateChecked(maxValue);

        return decimal.Min(maxValue, decimal.CreateChecked(randomDouble) + SmallestPositiveDecimal);
    }

    private static decimal SmallestPositiveDecimal { get; } = new decimal(1, 0, 0, false, 28);
}
