namespace Daylin.Utilities.Serialization;

/// <summary>
/// Configuration profiles for serialization.
/// </summary>
public enum SerializationProfile
{
    /// <summary>
    /// Default profile with standard serialization behavior.
    /// </summary>
    Default,

    /// <summary>
    /// Profile for persistent storage (files, databases).
    /// </summary>
    Persistence,

    /// <summary>
    /// Profile for data transfer (APIs, wire protocols).
    /// </summary>
    DataTransfer
}
