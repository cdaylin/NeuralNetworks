using System.ComponentModel;
using System.Linq.Expressions;
using System.Reflection;

namespace Daylin.Utilities.Observable.Properties;

/// <summary>
/// Responsible for creating observable proxies for properties.
/// </summary>
/// <remarks>
/// This class supports creating <see cref="IObservableProperty{T}"/> instances for both single properties and
/// nested property paths. Registered property providers are queried in sequence until one is able to produce an
/// observable property proxy.
/// <para>
/// When creating an observable proxy from a <see cref="PropertyPath"/>, the factory observes the path as a chained
/// sequence of observable properties. If any intermediate property in the path evaluates to <c>null</c>, the value
/// will evaluate to the specified fallback value. The default fallback value is <c>default(T)</c>, which may be
/// <c>null</c> even when type parameter "T" is a non-nullable reference type.
/// </para><para>
/// Validation is performed at construction time to ensure all properties are supported by at least one registered
/// provider and the declared type of the final property is assignable to type "T".
/// </para>
/// </remarks>
public sealed class ObservablePropertyFactory
{
    #region Static

    /// <summary>
    /// A globally accessible factory intended for general use.
    /// </summary>
    /// <remarks>
    /// Initially a single observable property provider is registered,
    /// <see cref="ObservableNotifyPropertyProvider"/>, which is capable of observing public properties on types that
    /// implement <see cref="INotifyPropertyChanged"/>. Additional providers can be registered via 
    /// <see cref="RegisterGlobalPropertyProviders"/>.
    /// </remarks>
    /// <seealso cref="RegisterGlobalPropertyProviders"/>
    public static ObservablePropertyFactory Global { get; } = new(new ObservableNotifyPropertyProvider());

    /// <summary>
    /// Appends additional observable property providers to the list of providers used by the global factory
    /// (<see cref="Global"/>).
    /// </summary>
    /// <param name="providers">
    /// Observable property providers.
    /// </param>
    /// <seealso cref="Global"/>
    public static void RegisterGlobalPropertyProviders(params IEnumerable<IObservablePropertyProvider> providers)
    {
        providers.ThrowIfNull();
        providers.ThrowIfContainsNull();

        Global.AppendPropertyProviders(providers);
    }

    #endregion

    #region Construction

    public ObservablePropertyFactory(params IEnumerable<IObservablePropertyProvider> propertyProviders)
    {
        propertyProviders.ThrowIfNull();
        propertyProviders.ThrowIfContainsNull();

        this.propertyProviders = propertyProviders.Distinct().ToArray();
    }

    #endregion

    #region Public

    public IObservableProperty<T> Observe<T>(object source, string propertyName)
    {
        source.ThrowIfNull();
        propertyName.ThrowIfNullOrEmpty();

        foreach (IObservablePropertyProvider provider in PropertyProviders)
        {
            if (provider.Create<T>(source, propertyName) is IObservableProperty<T> property)
                return property;
        }

        throw new NotSupportedException(
            "No registered observable property provider can create an observable property for "
                + $"{source.GetType().FullName}.{propertyName}.");
    }

    public IObservableProperty<T> Observe<T>(
        object source,
        PropertyPath path,
        T fallbackValue = default!)
    {
        source.ThrowIfNull();
        path.ThrowIfNull();

        return new ChainedPropertyProxy<T>(source, path, fallbackValue, this);
    }

    public IObservableProperty<T> Observe<T>(
        object source,
        Expression<Func<object, T>> propertyPathExpression,
        T fallbackValue = default!)
    {
        source.ThrowIfNull();
        propertyPathExpression.ThrowIfNull();

        return Observe(source, PropertyPath.FromExpression(propertyPathExpression), fallbackValue);
    }

    public IObservableProperty<T> Observe<T, TSource>(
        TSource source,
        Expression<Func<TSource, T>> propertyPathExpression,
        T fallbackValue = default!)
    {
        source.ThrowIfNull();
        propertyPathExpression.ThrowIfNull();

        return Observe(source, PropertyPath.FromExpression(propertyPathExpression), fallbackValue);
    }

    public IReadOnlyList<IObservablePropertyProvider> PropertyProviders => propertyProviders;

    #endregion

    #region Private

    // volatile for thread-safe replacement
    private volatile IReadOnlyList<IObservablePropertyProvider> propertyProviders;

    private object ProvidersLock { get; } = new object();

    // Note: This is private to allow additional providers to be registered with the Global instance without causing
    //  all instances to be mutable. If it were made public, immutable factory instances could not be created.
    private void AppendPropertyProviders(params IEnumerable<IObservablePropertyProvider> providers)
    {
        lock (ProvidersLock)
        {
            propertyProviders = propertyProviders.Concat(providers).Distinct().ToArray();
        }
    }

    /// <summary>
    /// Acts as a proxy to a property specified by a property path.
    /// </summary>
    /// <typeparam name="T">
    /// Type of the evaluated property path.
    /// </typeparam>
    private sealed class ChainedPropertyProxy<T> : ObservableProperty<T>
    {
        #region Construction

        public ChainedPropertyProxy(
            object source,
            PropertyPath path,
            T fallbackValue,
            ObservablePropertyFactory observablePropertyFactory)
            : this(source, path, observablePropertyFactory, fallbackValue)
        {
        }

        private ChainedPropertyProxy(
            object source,
            PropertyPath path,
            ObservablePropertyFactory observablePropertyFactory,
            T fallbackValue)
        {
            source.ThrowIfNull();
            path.ThrowIfNull();
            observablePropertyFactory.ThrowIfNull();

            ValidatePropertyPath(source.GetType(), path.PropertyNames, typeof(T), observablePropertyFactory.PropertyProviders);

            PropertyPath = path;
            ObservablePropertyFactory = observablePropertyFactory;
            FallbackValue = fallbackValue;

            ObservableProperties = CreateObservableProperties(source, PropertyPath.PropertyNames);
            Value = GetValue();
        }

        #endregion

        #region Public

        public PropertyPath PropertyPath { get; }

        public T FallbackValue { get; }

        #endregion

        #region Protected

        protected override void ReleaseResources()
        {
            foreach (IObservableProperty<object?> propertyObserver in ObservableProperties)
                propertyObserver.Dispose();

            ObservableProperties.Clear();

            base.ReleaseResources();
        }

        #endregion

        #region Private

        private List<IObservableProperty<object?>> ObservableProperties { get; }

        private ObservablePropertyFactory ObservablePropertyFactory { get; }

        private static void ValidatePropertyPath(
            Type sourceType,
            IEnumerable<string> propertyNames,
            Type expectedFinalType,
            IReadOnlyList<IObservablePropertyProvider> providers)
        {
            Type currentHostType = sourceType;
            string? lastProperty = null;

            foreach (string propertyName in propertyNames)
            {
                if (!IsPropertySupported(currentHostType, propertyName, providers))
                {
                    throw new NotSupportedException(
                        $"No registered observable property provider supports observing property "
                            + $"'{propertyName}' on type '{currentHostType.FullName}'.");
                }

                PropertyInfo? propertyInfo = currentHostType.GetProperty(
                    propertyName,
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

                if (propertyInfo is null)
                {
                    throw new InvalidOperationException(
                        $"Property '{propertyName}' does not exist on type '{currentHostType.FullName}'.");
                }

                lastProperty = propertyName;
                currentHostType = propertyInfo.PropertyType;
            }

            if (lastProperty != null && !expectedFinalType.IsAssignableFrom(currentHostType))
            {
                throw new InvalidOperationException(
                    $"The final property '{lastProperty}' on type '{currentHostType.FullName}' "
                        + $"is not assignable to '{expectedFinalType.FullName}'.");
            }
        }

        private static bool IsPropertySupported(
            Type sourceType,
            string propertyName,
            IReadOnlyList<IObservablePropertyProvider> providers)
        {
            return providers.Any(provider => provider.IsPropertySupported(sourceType, propertyName));
        }

        private List<IObservableProperty<object?>> CreateObservableProperties(
            object? source,
            IEnumerable<string> propertyNames)
        {
            List<IObservableProperty<object?>> properties = [];

            object? currentValue = source;

            foreach (string propertyName in propertyNames)
            {
                if (currentValue is null)
                {
                    properties.Add(NullPropertyObserver.Instance);
                    continue;
                }

                IObservableProperty<object?> property =
                    ObservablePropertyFactory.Observe<object?>(currentValue, propertyName);

                property.ValueChanged += HandlePropertyValueChanged;
                properties.Add(property);

                currentValue = property.Value;
            }

            return properties;
        }

        private void HandlePropertyValueChanged(object? sender, EventArgs args)
        {
            if (IsDisposed)
                return;

            int changedIndex = ObservableProperties.IndexOf((IObservableProperty<object?>)sender!);

            if (changedIndex < 0)
                return;

            for (int index = changedIndex + 1; index < ObservableProperties.Count; index++)
                ObservableProperties[index].Dispose();

            ObservableProperties.RemoveRange(changedIndex + 1, ObservableProperties.Count - changedIndex - 1);

            ObservableProperties.AddRange(CreateObservableProperties(
                ObservableProperties[changedIndex].Value,
                PropertyPath.PropertyNames.Skip(changedIndex + 1)));

            Value = GetValue();
        }

        private T GetValue()
        {
            object? finalValue = ObservableProperties.Last().Value;

            if (finalValue is null && HasIntermediateNull())
                return FallbackValue;

            return (T)finalValue!;
        }

        private bool HasIntermediateNull()
        {
            return ObservableProperties.OfType<NullPropertyObserver>().Any();
        }

        #endregion
    }

    private sealed class NullPropertyObserver : IObservableProperty<object?>
    {
        /// <summary>
        /// Singleton instance.
        /// </summary>
        public static NullPropertyObserver Instance { get; } = new NullPropertyObserver();

        private NullPropertyObserver()
        {
        }

        public object? Value => null;

        public event PropertyChangedEventHandler? PropertyChanged { add { } remove { } }

        public event EventHandler<PropertyChangedEventArgs<object?>>? ValueChanged { add { } remove { } }

        public void Dispose() { }
    }

    #endregion
}
