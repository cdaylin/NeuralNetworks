using Daylin.Utilities.Comparisons;

using System.Collections.Concurrent;
using System.Linq.Expressions;
using System.Reflection;

namespace Daylin.Utilities.Comparers;

/// <summary>
/// Provides factory methods for constructing <see cref="IComparer{T}"/> instances.
/// </summary>
/// <remarks>
/// Constructed comparers are cached to improve efficiency for subsequent requests.
/// </remarks>
public static class ComparerFactory
{
    /// <summary>
    /// Creates an <see cref="IComparer{T}"/> that compares instances by a prioritized list of properties.
    /// </summary>
    /// <remarks>
    /// The comparer sorts <c>null</c> items before non-<c>null</c> items. Then non-<c>null</c> items are
    /// sorted by comparing the values of specified properties.
    /// </remarks>
    /// <typeparam name="T">
    /// Type of items to compare.
    /// </typeparam>
    /// <param name="propertyNames">
    /// Names of properties to compare, in priority order.
    /// </param>
    /// <returns>
    /// Returns an <see cref="IComparer{T}"/> for comparing items by the specified properties.
    /// </returns>
    public static IComparer<T?> CreateComparer<T>(params IReadOnlyList<string> propertyNames)
    {
        propertyNames.ThrowIfNull();
        propertyNames.ThrowIfContainsNull();

        if (propertyNames.Count == 0)
            return new NullComparer<T>();

        string cacheKey = string.Join("\0", propertyNames);

        if (Cache.TryGetValue((typeof(T), cacheKey), out object? cachedComparer))
            return (IComparer<T?>)cachedComparer;

        ParameterExpression leftParameter = Expression.Parameter(typeof(T), "left");
        ParameterExpression rightParameter = Expression.Parameter(typeof(T), "right");

        ParameterExpression resultVariable = Expression.Variable(typeof(int), "comparisonResult");
        LabelTarget returnLabel = Expression.Label(typeof(int));

        List<Expression> expressionBlock = [];

        foreach (string propertyName in propertyNames)
        {
            PropertyInfo? propertyInfo = typeof(T).GetProperty(
                propertyName,
                BindingFlags.Instance | BindingFlags.Public);

            if (propertyInfo is null)
                throw new ArgumentException($"Property '{propertyName}' not found on type '{typeof(T).Name}'.");

            Expression leftProperty = Expression.Property(leftParameter, propertyInfo);
            Expression rightProperty = Expression.Property(rightParameter, propertyInfo);

            Type comparerType = typeof(Comparer<>).MakeGenericType(propertyInfo.PropertyType);

            MethodInfo compareMethod = comparerType.GetMethod(
                "Compare", new[] { propertyInfo.PropertyType, propertyInfo.PropertyType })!;

            PropertyInfo defaultComparerProperty = comparerType.GetProperty("Default")!;

            Expression defaultComparer = Expression.Property(null, defaultComparerProperty);
            Expression compareCall = Expression.Call(defaultComparer, compareMethod, leftProperty, rightProperty);

            expressionBlock.Add(Expression.Assign(resultVariable, compareCall));
            expressionBlock.Add(
                Expression.IfThen(
                    Expression.NotEqual(resultVariable, Expression.Constant(0)),
                    Expression.Return(returnLabel, resultVariable)));
        }

        expressionBlock.Add(Expression.Label(returnLabel, Expression.Constant(0)));

        BlockExpression blockExpression = Expression.Block(new[] { resultVariable }, expressionBlock);

        Comparison<T> comparison = Expression
            .Lambda<Comparison<T>>(blockExpression, leftParameter, rightParameter)
            .Compile();

        IComparer<T?> comparer = new NullComparer<T>(Comparer<T>.Create(comparison));

        Cache[(typeof(T), cacheKey)] = comparer;

        return comparer;
    }

    private static readonly ConcurrentDictionary<(Type, string), object> Cache = new();
}
