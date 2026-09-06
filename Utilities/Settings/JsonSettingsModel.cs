using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;

using Daylin.Utilities.Observable;
using Daylin.Utilities.Serialization;
using Daylin.Utilities.Serialization.Json;

namespace Daylin.Utilities.Settings;

/// <summary>
/// Base class for settings models backed by a JSON file.
/// Properties are stored as key-value pairs and auto-saved on change.
/// </summary>
public abstract class JsonSettingsModel : ObservableObject
{
    #region Construction

    /// <summary>
    /// Creates a new JSON settings model.
    /// </summary>
    /// <param name="filePath">Path to the JSON file.</param>
    protected JsonSettingsModel(string filePath)
    {
        filePath.ThrowIfNullOrEmpty();

        FilePath = filePath;

        LoadFromFile();
    }

    #endregion

    #region Protected

    /// <summary>
    /// Gets the value of a property from the backing store.
    /// </summary>
    /// <typeparam name="T">Property type.</typeparam>
    /// <param name="defaultValue">Default value if the property is not set.</param>
    /// <param name="propertyName">Property name (auto-populated by compiler).</param>
    /// <returns>The property value or default.</returns>
    protected T GetBackingProperty<T>(
        T defaultValue = default!,
        [CallerMemberName] string propertyName = "")
    {
        if (!Values.TryGetPropertyValue(propertyName, out JsonNode? node) || node is null)
            return defaultValue;

        try
        {
            return node.Deserialize<T>(Json.CreateOptions(SerializationProfile.Persistence)) ?? defaultValue;
        }
        catch (JsonException)
        {
            return defaultValue;
        }
    }

    /// <summary>
    /// Sets the value of a property in the backing store.
    /// </summary>
    /// <typeparam name="T">Property type.</typeparam>
    /// <param name="currentValue">Current value (for equality comparison).</param>
    /// <param name="newValue">New value to set.</param>
    /// <param name="propertyName">Property name (auto-populated by compiler).</param>
    /// <returns>True if the value was changed.</returns>
    protected bool SetBackingProperty<T>(
        T currentValue,
        T newValue,
        [CallerMemberName] string propertyName = "")
    {
        if (Equals(currentValue, newValue))
            return false;

        Values[propertyName] = Json.SerializeToNode(newValue, SerializationProfile.Persistence);
        SaveToFile();
        RaisePropertyChangedEvent(propertyName);

        return true;
    }

    /// <summary>
    /// Gets a Guid property value.
    /// </summary>
    protected Guid? GetGuidProperty([CallerMemberName] string propertyName = "")
    {
        string? storedValue = GetBackingProperty<string?>(null, propertyName);

        if (string.IsNullOrEmpty(storedValue))
            return null;

        return Guid.TryParse(storedValue, out Guid result) ? result : null;
    }

    /// <summary>
    /// Sets a Guid property value.
    /// </summary>
    protected void SetGuidProperty(Guid? value, [CallerMemberName] string propertyName = "")
    {
        string? currentStored = GetBackingProperty<string?>(null, propertyName);
        string? newStored = value?.ToString();

        SetBackingProperty(currentStored, newStored, propertyName);
    }

    #endregion

    #region Private

    private string FilePath { get; }

    private JsonObject Values { get; set; } = [];

    private void LoadFromFile()
    {
        if (!File.Exists(FilePath))
        {
            Values = [];
            return;
        }

        try
        {
            string json = File.ReadAllText(FilePath);
            Values = JsonNode.Parse(json)?.AsObject() ?? [];
        }
        catch (JsonException)
        {
            Values = [];
        }
        catch (IOException)
        {
            Values = [];
        }
    }

    private void SaveToFile()
    {
        try
        {
            string? directory = Path.GetDirectoryName(FilePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                Directory.CreateDirectory(directory);

            string json = Values.ToJsonString(Json.CreateOptions(SerializationProfile.Persistence));
            File.WriteAllText(FilePath, json);
        }
        catch (IOException)
        {
            // Silently fail on save errors - settings will be lost but app continues
        }
    }

    #endregion
}
