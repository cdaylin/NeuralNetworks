using System.Diagnostics;

namespace Daylin.Utilities.Threading;

/// <summary>
/// A wrapper around a <c>bool</c> variable that allows the variable to be read and updated as an atomic
/// operation.
/// </summary>
[DebuggerDisplay("{Value}")]
public sealed class InterlockedBool
{
    #region Construction

    /// <summary>
    /// Constructs a <see cref="InterlockedBool"/> object with an initial value of <c>false</c>.
    /// </summary>
    public InterlockedBool()
        : this(false)
    {
    }

    /// <summary>
    /// Constructs a <see cref="InterlockedBool"/> object with a specified initial value.
    /// </summary>
    /// <param name="value">
    /// Initial value.
    /// </param>
    public InterlockedBool(bool value)
    {
        valueAsInt = ToInt(value);
    }

    #endregion

    #region Public

    /// <summary>
    /// Reads and sets the value as an atomic operation.
    /// </summary>
    /// <param name="value">
    /// The new value.
    /// </param>
    /// <returns>
    /// Returns <c>true</c> if and only if the value was modified.  Returns <c>false</c> if no change
    /// was made because the value was already equal to <paramref name="value"/>.
    /// </returns>
    public bool TrySetValue(bool value)
    {
        return value != ToBool(Interlocked.Exchange(ref valueAsInt, ToInt(value)));
    }

    /// <summary>
    /// The Boolean value.
    /// </summary>
    public bool Value => ToBool(valueAsInt);

    public bool IsTrue => ToBool(valueAsInt);

    public bool IsFalse => !IsTrue;

    #endregion

    #region Private

    private volatile int valueAsInt;

    private static int FalseAsInt => 0;

    private static int TrueAsInt => 1;

    private static int ToInt(bool value) => value ? TrueAsInt : FalseAsInt;

    private static bool ToBool(int value) => value == TrueAsInt;

    #endregion
}
