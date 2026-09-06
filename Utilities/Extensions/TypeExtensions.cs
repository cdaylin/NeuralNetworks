using System.Reflection;

namespace Daylin.Utilities.Extensions;

/// <summary>
/// Extension methods for class <see cref="Type"/>.
/// </summary>
public static class TypeExtensions
{
    public static IEnumerable<PropertyInfo> GetPublicNonindexedInstanceProperties(this Type @type)
    {
        @type.ThrowIfNull();

        return @type
            .GetProperties(BindingFlags.Instance | BindingFlags.Public)
            .Where(propertyInfo => propertyInfo.GetIndexParameters().IsNullOrEmpty());
    }
}
