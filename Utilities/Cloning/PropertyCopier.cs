using System.Linq.Expressions;
using System.Reflection;

namespace Daylin.Utilities.Cloning;

/// <summary>
/// Responsible for copying values of public settable properties from one object to another object of the same type.
/// </summary>
/// <typeparam name="T">
/// Type of object for which properties will be copied. Properties to be copied are selected based on this type, not
/// the runtime type of the objects passed to <see cref="CopyProperties(T, T, bool)"/>.
/// </typeparam>
public static class PropertyCopier<T>
{
    #region Public Static

    /// <summary>
    /// Copies the values of public settable properties from a source object to a target object.
    /// </summary>
    /// <param name="source">
    /// The source object from which properties will be copied.
    /// </param>
    /// <param name="target">
    /// The target object to which properties will be copied.
    /// </param>
    /// <param name="declaredOnly">
    /// When <c>true</c>, only properties declared on type <typeparamref name="T"/> are copied.
    /// When <c>false</c> (default), inherited properties are included.
    /// </param>
    public static void CopyProperties(T source, T target, bool declaredOnly = false)
    {
        source.ThrowIfNull();
        target.ThrowIfNull();

        if (declaredOnly)
            CopyDeclaredOnlyAction(source, target);
        else
            CopyAllAction(source, target);
    }

    #endregion

    #region Private Static

    private static Action<T, T> CopyAllAction { get; } = BuildCopyAction(declaredOnly: false);

    private static Action<T, T> CopyDeclaredOnlyAction { get; } = BuildCopyAction(declaredOnly: true);

    private static Action<T, T> BuildCopyAction(bool declaredOnly)
    {
        ParameterExpression sourceParam = Expression.Parameter(typeof(T), "source");
        ParameterExpression targetParam = Expression.Parameter(typeof(T), "target");

        List<Expression> assignments = [];

        BindingFlags bindingFlags = BindingFlags.Public | BindingFlags.Instance;

        if (declaredOnly)
            bindingFlags |= BindingFlags.DeclaredOnly;

        foreach (PropertyInfo property in typeof(T).GetProperties(bindingFlags))
        {
            if (!property.CanRead || !property.CanWrite)
                continue;

            MemberExpression sourceValue = Expression.Property(sourceParam, property);
            MemberExpression targetProperty = Expression.Property(targetParam, property);
            BinaryExpression assign = Expression.Assign(targetProperty, sourceValue);

            assignments.Add(assign);
        }

        BlockExpression body = Expression.Block(assignments);
        Expression<Action<T, T>> lambda = Expression.Lambda<Action<T, T>>(body, sourceParam, targetParam);

        return lambda.Compile();
    }

    #endregion
}
