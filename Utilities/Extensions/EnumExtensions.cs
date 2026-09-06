using System.ComponentModel.DataAnnotations;
using System.Reflection;

namespace Daylin.Utilities.Extensions;

/// <summary>
/// Extension methods for type <see cref="Enum"/>.
/// </summary>
public static class EnumExtensions
{
    /// <summary>
    /// Gets the display name for an enum value.
    /// </summary>
    /// <remarks>
    /// The display name is obtained from <see cref="DisplayAttribute.Name"/>, if that attribute is applied
    /// to the enum constant and the name is not <c>null</c>.  Otherwise, the text of the enum constant is used.
    /// </remarks>
    /// <param name="enumValue">
    /// An enum value.
    /// </param>
    /// <returns>
    /// Returns the display name for <paramref name="enumValue"/>.
    /// </returns>
    public static string GetDisplayName(this Enum enumValue)
    {
        enumValue.ThrowIfNull();

        if (GetAttribute<DisplayAttribute>(enumValue) is DisplayAttribute displayAttribute)
        {
            if (displayAttribute.Name is not null)
                return displayAttribute.Name;
        }

        return enumValue.ToString();
    }

    /// <summary>
    /// Gets the short display name for an enum value.
    /// </summary>
    /// <remarks>
    /// The short name is obtained from <see cref="DisplayAttribute.ShortName"/>, if that attribute is applied
    /// to the enum constant and the short name is not <c>null</c>.  Otherwise, the text of the enum constant is
    /// used.
    /// </remarks>
    /// <param name="enumValue">
    /// An enum value.
    /// </param>
    /// <returns>
    /// Returns the short display name for <paramref name="enumValue"/>.
    /// </returns>
    public static string GetShortName(this Enum enumValue)
    {
        enumValue.ThrowIfNull();

        if (GetAttribute<DisplayAttribute>(enumValue) is DisplayAttribute displayAttribute)
        {
            if (displayAttribute.ShortName is not null)
                return displayAttribute.ShortName;
        }

        return enumValue.ToString();
    }

    /// <summary>
    /// Gets the description for an enum value.
    /// </summary>
    /// <remarks>
    /// The description is obtained from <see cref="DisplayAttribute.Description"/>, if that attribute is applied
    /// to the enum constant and the description is not <c>null</c>.  Otherwise, the result of 
    /// <see cref="GetDisplayName"/> is used.
    /// </remarks>
    /// <param name="enumValue">
    /// An enum value.
    /// </param>
    /// <returns>
    /// Returns the description for <paramref name="enumValue"/>.
    /// </returns>
    public static string GetDescription(this Enum enumValue)
    {
        enumValue.ThrowIfNull();

        if (GetAttribute<DisplayAttribute>(enumValue) is DisplayAttribute displayAttribute)
        {
            if (displayAttribute.Description is not null)
                return displayAttribute.Description;
        }

        return enumValue.GetDisplayName();
    }

    public static T? GetAttribute<T>(this Enum enumValue)
        where T : Attribute
    {
        enumValue.ThrowIfNull();

        return enumValue
            .GetAttributes<T>()
            .FirstOrDefault();
    }

    public static IEnumerable<T> GetAttributes<T>(this Enum enumValue)
        where T : Attribute
    {
        enumValue.ThrowIfNull();

        return enumValue
            .GetType()
            .GetMember(enumValue.ToString())
            .First()
            .GetCustomAttributes<T>();
    }
}