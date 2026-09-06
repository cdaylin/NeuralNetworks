using System.Diagnostics;
using System.Linq.Expressions;
using System.Reflection;

namespace Daylin.Utilities.Observable.Properties;

/// <summary>
/// Represents a path of property names for accessing a nested property.
/// </summary>
[DebuggerDisplay("{ToString(),nq}")]
public sealed class PropertyPath
{
    #region Static

    /// <summary>
    /// Creates a property path from a strongly-typed property access expression.
    /// </summary>
    /// <typeparam name="TRoot">
    /// Type of the root object containing the property path.
    /// </typeparam>
    /// <typeparam name="T">
    /// Type of the final property in the path.
    /// </typeparam>
    /// <param name="propertyPathExpression">
    /// A lambda expression that specifies the property path to observe.  For example: <c>x => x.Address.City</c>
    /// will result in a path of ["Address", "City"].
    /// </param>
    /// <returns>
    /// A <see cref="PropertyPath"/> instance representing the sequence of property names in the expression.
    /// </returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="propertyPathExpression"/> is <c>null</c>.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the expression does not represent a simple property path (e.g., contains method calls, indexers,
    /// fields, etc.).
    /// </exception>
    public static PropertyPath FromExpression<TRoot, T>(Expression<Func<TRoot, T>> propertyPathExpression)
    {
        propertyPathExpression.ThrowIfNull();

        List<string> segments = [];
        Expression? current = StripConvert(propertyPathExpression.Body);

        while (current is MemberExpression member)
        {
            if (member.Member is not PropertyInfo propertyInfo)
            {
                throw new ArgumentException(
                    "Expression must access properties, not fields.",
                    nameof(propertyPathExpression));
            }

            if (propertyInfo.GetMethod?.IsStatic == true)
            {
                throw new ArgumentException(
                    "Static properties are not supported in a property path.",
                    nameof(propertyPathExpression));
            }

            segments.Insert(0, propertyInfo.Name);

            // Note that Expression will not be null for a non-static property
            current = StripConvert(member.Expression!);
        }

        if (current is ParameterExpression)
        {
            if (segments.Count == 0)
            {
                throw new ArgumentException(
                    "Expression must select at least one property (e.g., x => x.Property).",
                    nameof(propertyPathExpression));
            }

            return new PropertyPath(segments.ToArray());
        }

        if (current is MethodCallExpression)
        {
            throw new ArgumentException(
                "Method calls are not supported in a property path.",
                nameof(propertyPathExpression));
        }

        if (current is IndexExpression)
        {
            throw new ArgumentException(
                "Indexers are not supported in a property path.",
                nameof(propertyPathExpression));
        }

        if (current is ConditionalExpression)
        {
            throw new ArgumentException(
                "Conditional operators are not supported in a property path.",
                nameof(propertyPathExpression));
        }

        // Fallback: unknown/complex node
        throw new ArgumentException(
            $"Expression is not a simple property path. Offending node: {current?.NodeType.ToString() ?? "null"}.",
            nameof(propertyPathExpression));
    }

    private static Expression? StripConvert(Expression? expression)
    {
        Expression? current = expression;

        while (current is UnaryExpression unaryExpression
            && (unaryExpression.NodeType == ExpressionType.Convert
                || unaryExpression.NodeType == ExpressionType.ConvertChecked))
            current = unaryExpression.Operand;

        return current;
    }

    #endregion

    #region Construction

    public PropertyPath(params IReadOnlyList<string> propertyNames)
    {
        propertyNames.ThrowIfNull();
        propertyNames.ThrowIfEmpty();
        propertyNames.ThrowIfContainsNull();

        PropertyNames = propertyNames.ToArray();
    }

    #endregion

    #region Public

    /// <summary>
    /// The list of property names that form the path.
    /// </summary>
    public IReadOnlyList<string> PropertyNames { get; }

    public override string ToString()
    {
        return string.Join(".", PropertyNames);
    }

    #endregion
}
