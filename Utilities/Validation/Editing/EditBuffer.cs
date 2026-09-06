using Daylin.Utilities.Commands;
using Daylin.Utilities.Observable;

using System.Collections;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Reflection;
using System.Windows.Input;

namespace Daylin.Utilities.Validation.Editing;

public class EditBuffer : ObservableObject
{
    #region Construction

    public EditBuffer(IEditBufferCloneSource entity, bool isNew = false)
    {
        entity.ThrowIfNull();

        Source = entity;
        Entity = entity.CreateEditBufferClone();

        PropertyTrackers = CreatePropertyTrackers(entity);

        foreach (PropertyTracker tracker in PropertyTrackers.Values)
            tracker.Initialize();

        IsNew = isNew;

        ApplyChangesCommand = new SaveCommand(this);
        RevertChangesCommand = new CancelCommand(this);

        Source.PropertyChanged += HandleSourcePropertyChanged;
        Entity.PropertyChanged += HandleEntityPropertyChanged;
        Entity.ErrorNotifier.PropertyChanged += HandleErrorNotifierPropertyChanged;

        UpdateHasChanges();
        UpdateHasErrors();
    }

    #endregion

    #region Public

    public event EventHandler? ChangesApplied;

    public event EventHandler? ChangesReverted;

    public IValidatable Entity { get; }

    public IValidatable Source { get; }

    public bool IsNew { get; private set => SetProperty(ref field, value); }

    public bool HasChanges { get; private set => SetProperty(ref field, value); }

    public bool HasErrors { get; private set => SetProperty(ref field, value); }

    public void ApplyChanges()
    {
        if (!HasChanges)
            return;

        foreach (PropertyTracker tracker in PropertyTrackers.Values)
        {
            if (!tracker.IsModified)
                continue;

            tracker.ApplyChanges();
            tracker.RevertChanges();
        }

        IsNew = false;
        UpdateHasChanges();

        ChangesApplied?.Invoke(this, EventArgs.Empty);
    }

    public void RevertChanges()
    {
        if (!HasChanges)
            return;

        foreach (PropertyTracker tracker in PropertyTrackers.Values)
        {
            if (tracker.IsModified)
                tracker.RevertChanges();
        }

        UpdateHasChanges();

        ChangesReverted?.Invoke(this, EventArgs.Empty);
    }

    public ICommand ApplyChangesCommand { get; }

    public ICommand RevertChangesCommand { get; }

    #endregion

    #region Protected

    protected override void ReleaseResources()
    {
        Source.PropertyChanged -= HandleSourcePropertyChanged;
        Entity.PropertyChanged -= HandleEntityPropertyChanged;
        Entity.ErrorNotifier.PropertyChanged -= HandleErrorNotifierPropertyChanged;

        foreach (PropertyTracker tracker in PropertyTrackers.Values)
            tracker.Dispose();

        if (Entity is IDisposable disposableBuffer)
            disposableBuffer.Dispose();

        base.ReleaseResources();
    }

    #endregion

    #region Private

    private Dictionary<string, PropertyTracker> PropertyTrackers { get; }

    private void UpdateHasChanges()
    {
        HasChanges = IsNew || PropertyTrackers.Values.Any(tracker => tracker.IsModified);
    }

    private void UpdateHasErrors()
    {
        HasErrors = Entity.ErrorNotifier.HasErrors;
    }

    private Dictionary<string, PropertyTracker> CreatePropertyTrackers(IEditBufferCloneSource entity)
    {
        Dictionary<string, PropertyTracker> trackers = new(StringComparer.Ordinal);

        foreach (PropertyInfo propertyInfo in entity
            .GetType()
            .GetProperties(BindingFlags.Instance | BindingFlags.Public))
        {
            PropertyTracker? tracker = CreatePropertyTracker(propertyInfo);

            if (tracker is null)
                continue;

            trackers[propertyInfo.Name] = tracker;
        }

        return trackers;
    }

    private PropertyTracker? CreatePropertyTracker(PropertyInfo propertyInfo)
    {
        if (!propertyInfo.CanRead)
            return null;

        object? sourceValue = propertyInfo.GetValue(Source);
        object? entityValue = propertyInfo.GetValue(Entity);

        if (IsCollectionProperty(propertyInfo, sourceValue, entityValue))
            return new CollectionPropertyTracker(this, propertyInfo);

        if (propertyInfo.SetMethod is { IsPublic: true })
            return new ScalarPropertyTracker(this, propertyInfo);

        return null;
    }

    private static bool IsCollectionProperty(PropertyInfo propertyInfo, object? sourceValue, object? entityValue)
    {
        Type propertyType = propertyInfo.PropertyType;

        if (propertyType == typeof(string))
            return false;

        if (!typeof(IEnumerable).IsAssignableFrom(propertyType))
            return false;

        if (IsDeclaredAsReadOnlyCollectionType(propertyType))
            return false;

        return sourceValue is IEnumerable || entityValue is IEnumerable;
    }

    private static bool IsDeclaredAsReadOnlyCollectionType(Type propertyType)
    {
        if (propertyType == typeof(IEnumerable))
            return true;

        if (propertyType.IsInterface && propertyType.IsGenericType)
        {
            Type definition = propertyType.GetGenericTypeDefinition();

            if (definition == typeof(IEnumerable<>) ||
                definition == typeof(IReadOnlyCollection<>) ||
                definition == typeof(IReadOnlyList<>))
            {
                return true;
            }
        }

        if (propertyType.IsGenericType)
        {
            Type definition = propertyType.GetGenericTypeDefinition();

            if (definition == typeof(ReadOnlyCollection<>) ||
                definition == typeof(ReadOnlyObservableCollection<>))
            {
                return true;
            }
        }

        return false;
    }

    private void HandleErrorNotifierPropertyChanged(object? sender, PropertyChangedEventArgs args)
    {
        if (args.PropertyName is nameof(IObservableErrorNotifier.HasErrors))
            UpdateHasErrors();
    }

    private void HandleSourcePropertyChanged(object? sender, PropertyChangedEventArgs args)
    {
        if (args.PropertyName is null)
            return;

        if (!PropertyTrackers.TryGetValue(args.PropertyName, out PropertyTracker? tracker))
            return;

        tracker.HandleSourcePropertyChanged();

        UpdateHasChanges();
    }

    private void HandleEntityPropertyChanged(object? sender, PropertyChangedEventArgs args)
    {
        if (args.PropertyName is null)
            return;

        if (!PropertyTrackers.TryGetValue(args.PropertyName, out PropertyTracker? tracker))
            return;

        tracker.HandleEntityPropertyChanged();

        UpdateHasChanges();
    }

    private interface IPropertyTracker
    {
        bool IsModified { get; }

        void ApplyChanges();

        void RevertChanges();
    }

    private abstract class PropertyTracker : IPropertyTracker, IDisposable
    {
        protected PropertyTracker(EditBuffer owner, PropertyInfo property)
        {
            Owner = owner;
            Property = property;
        }

        public abstract bool IsModified { get; }

        public abstract void ApplyChanges();

        public abstract void RevertChanges();

        public virtual void Initialize()
        {
        }

        public virtual void HandleEntityPropertyChanged()
        {
        }

        public virtual void HandleSourcePropertyChanged()
        {
            RevertChanges();
        }

        public virtual void Dispose()
        {
        }

        protected EditBuffer Owner { get; }

        protected PropertyInfo Property { get; }

        protected bool HasSetter => Property.SetMethod is { IsPublic: true };

        protected object? GetEntityValue() => Property.GetValue(Owner.Entity);

        protected object? GetSourceValue() => Property.GetValue(Owner.Source);

        protected void SetEntityValue(object? value)
        {
            if (!HasSetter)
                throw new InvalidOperationException($"Property '{Property.Name}' does not have a public setter.");

            Property.SetValue(Owner.Entity, value);
        }

        protected void SetSourceValue(object? value)
        {
            if (!HasSetter)
                throw new InvalidOperationException($"Property '{Property.Name}' does not have a public setter.");

            Property.SetValue(Owner.Source, value);
        }
    }

    private sealed class ScalarPropertyTracker : PropertyTracker
    {
        public ScalarPropertyTracker(EditBuffer owner, PropertyInfo property)
            : base(owner, property)
        {
        }

        public override bool IsModified
        {
            get
            {
                object? entityValue = GetEntityValue();
                object? sourceValue = GetSourceValue();

                return !Equals(entityValue, sourceValue);
            }
        }

        public override void ApplyChanges()
        {
            SetSourceValue(GetEntityValue());
        }

        public override void RevertChanges()
        {
            SetEntityValue(GetSourceValue());
        }
    }

    private sealed class CollectionPropertyTracker : PropertyTracker
    {
        public CollectionPropertyTracker(EditBuffer owner, PropertyInfo property)
            : base(owner, property)
        {
            equalityComparer = EqualityComparer<object?>.Default;
        }

        public override bool IsModified
        {
            get
            {
                IEnumerable? entityItems = GetEntityValue() as IEnumerable;
                IEnumerable? sourceItems = GetSourceValue() as IEnumerable;

                if (ReferenceEquals(entityItems, sourceItems))
                    return false;

                if (entityItems is null || sourceItems is null)
                    return entityItems is not null || sourceItems is not null;

                return !entityItems.Cast<object?>().SequenceEqual(
                    sourceItems.Cast<object?>(),
                    equalityComparer);
            }
        }

        public override void Initialize()
        {
            EnsureSupportedCollection(GetEntityValue());
            EnsureSupportedCollection(GetSourceValue());

            UpdateSubscriptions();

            if (IsModified)
                RevertChanges();
        }

        public override void ApplyChanges()
        {
            if (!IsModified)
                return;

            CopyCollection(GetEntityValue(), GetSourceValue(), copyToSource: true);
        }

        public override void RevertChanges()
        {
            CopyCollection(GetSourceValue(), GetEntityValue(), copyToSource: false);
        }

        public override void HandleEntityPropertyChanged()
        {
            UpdateSubscriptions();
        }

        public override void HandleSourcePropertyChanged()
        {
            UpdateSubscriptions();
            base.HandleSourcePropertyChanged();
        }

        public override void Dispose()
        {
            DetachSourceSubscription();
            DetachEntitySubscription();

            base.Dispose();
        }

        private void CopyCollection(object? source, object? target, bool copyToSource)
        {
            using CollectionCopyScope scope = new(this);

            if (source is null)
            {
                if (target is null)
                {
                    AssignNullIfPossible(copyToSource);
                    UpdateSubscriptions();
                    return;
                }

                if (!TryClearCollection(target))
                    AssignNullIfPossible(copyToSource);

                UpdateSubscriptions();
                return;
            }

            if (target is null)
            {
                if (!HasSetter)
                    throw new InvalidOperationException(
                        $"Collection property '{Property.Name}' cannot be assigned because it does not expose a public setter.");

                IList newCollection = CreateCollectionInstance(source);
                AssignCollectionInstance(copyToSource, newCollection);
                UpdateSubscriptions();
                return;
            }

            EnsureSupportedCollection(target);

            if (!ReplaceContents((IEnumerable)source, target))
                throw new InvalidOperationException(
                    $"Collection property '{Property.Name}' must support clearing and adding items.");

            UpdateSubscriptions();
        }

        private void AssignCollectionInstance(bool assignToSource, IList collection)
        {
            if (assignToSource)
                SetSourceValue(collection);
            else
                SetEntityValue(collection);
        }

        private void AssignNullIfPossible(bool assignToSource)
        {
            if (!HasSetter)
                return;

            if (assignToSource)
                SetSourceValue(null);
            else
                SetEntityValue(null);
        }

        private void UpdateSubscriptions()
        {
            DetachEntitySubscription();
            DetachSourceSubscription();

            AttachEntitySubscription(GetEntityValue());
            AttachSourceSubscription(GetSourceValue());
        }

        private void AttachEntitySubscription(object? value)
        {
            entityCollection = value as INotifyCollectionChanged;

            if (entityCollection is not null)
                entityCollection.CollectionChanged += HandleEntityCollectionChanged;
        }

        private void AttachSourceSubscription(object? value)
        {
            sourceCollection = value as INotifyCollectionChanged;

            if (sourceCollection is not null)
                sourceCollection.CollectionChanged += HandleSourceCollectionChanged;
        }

        private void DetachEntitySubscription()
        {
            if (entityCollection is not null)
                entityCollection.CollectionChanged -= HandleEntityCollectionChanged;

            entityCollection = null;
        }

        private void DetachSourceSubscription()
        {
            if (sourceCollection is not null)
                sourceCollection.CollectionChanged -= HandleSourceCollectionChanged;

            sourceCollection = null;
        }

        private void HandleEntityCollectionChanged(object? sender, NotifyCollectionChangedEventArgs args)
        {
            if (isCopying)
                return;

            Owner.UpdateHasChanges();
        }

        private void HandleSourceCollectionChanged(object? sender, NotifyCollectionChangedEventArgs args)
        {
            if (isCopying)
                return;

            RevertChanges();
            Owner.UpdateHasChanges();
        }

        private void EnsureSupportedCollection(object? value)
        {
            if (value is null)
                return;

            if (value is IList)
                return;

            MethodInfo? clearMethod = GetClearMethod(value);
            MethodInfo? addMethod = GetAddMethod(value);

            if (clearMethod is not null && addMethod is not null)
                return;

            throw new InvalidOperationException(
                $"Collection property '{Property.Name}' must support clearing and adding items.");
        }

        private static bool TryClearCollection(object collection)
        {
            if (collection is IList list)
            {
                list.Clear();
                return true;
            }

            MethodInfo? clearMethod = GetClearMethod(collection);

            if (clearMethod is null)
                return false;

            clearMethod.Invoke(collection, null);
            return true;
        }

        private IList CreateCollectionInstance(object source)
        {
            Type collectionType = Property.PropertyType;

            if (collectionType.IsInterface || collectionType.IsAbstract)
                collectionType = source.GetType();

            if (!typeof(IList).IsAssignableFrom(collectionType))
            {
                throw new InvalidOperationException(
                    $"Collection property '{Property.Name}' must implement IList to be cloned.");
            }

            if (collectionType.GetConstructor(Type.EmptyTypes) is null)
            {
                throw new InvalidOperationException(
                    $"Collection property '{Property.Name}' cannot be cloned because its type lacks a parameterless constructor.");
            }

            IList collection = (IList)Activator.CreateInstance(collectionType)!;

            foreach (object? item in (IEnumerable)source)
                collection.Add(item);

            return collection;
        }

        private static bool ReplaceContents(IEnumerable source, object target)
        {
            if (target is IList targetList)
            {
                targetList.Clear();

                foreach (object? item in source)
                    targetList.Add(item);

                return true;
            }

            MethodInfo? clearMethod = GetClearMethod(target);
            MethodInfo? addMethod = GetAddMethod(target);

            if (clearMethod is null || addMethod is null)
                return false;

            clearMethod.Invoke(target, null);

            foreach (object? item in source)
                addMethod.Invoke(target, new[] { item });

            return true;
        }

        private static MethodInfo? GetClearMethod(object collection)
        {
            return collection
                .GetType()
                .GetMethod(
                    "Clear",
                    BindingFlags.Instance | BindingFlags.Public, binder: null, Type.EmptyTypes, modifiers: null);
        }

        private static MethodInfo? GetAddMethod(object collection)
        {
            return collection
                .GetType()
                .GetMethods(BindingFlags.Instance | BindingFlags.Public)
                .FirstOrDefault(method =>
                {
                    if (!string.Equals(method.Name, "Add", StringComparison.Ordinal))
                        return false;

                    return method.GetParameters().Length == 1;
                });
        }

        private sealed class CollectionCopyScope : IDisposable
        {
            public CollectionCopyScope(CollectionPropertyTracker tracker)
            {
                this.tracker = tracker;
                previous = tracker.isCopying;
                tracker.isCopying = true;
            }

            public void Dispose()
            {
                tracker.isCopying = previous;
            }

            private readonly CollectionPropertyTracker tracker;
            private readonly bool previous;
        }

        private readonly IEqualityComparer<object?> equalityComparer;

        private INotifyCollectionChanged? entityCollection;

        private INotifyCollectionChanged? sourceCollection;

        private bool isCopying;
    }

    private class SaveCommand : Command
    {
        public SaveCommand(EditBuffer editBuffer)
        {
            EditBuffer = editBuffer;

            EditBuffer.PropertyChanged += HandleEditBufferPropertyChanged;
            EditBuffer.Entity.ErrorNotifier.ErrorsChanged += HandleBufferErrorsChanged;
        }

        public override bool CanExecute(object? parameter)
        {
            return EditBuffer.HasChanges && !EditBuffer.Entity.ErrorNotifier.HasErrors;
        }

        public override void Execute(object? parameter)
        {
            if (!CanExecute(parameter))
                return;

            EditBuffer.ApplyChanges();
        }

        private void HandleEditBufferPropertyChanged(object? sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(EditBuffer.HasChanges):
                    RaiseCanExecuteChanged();
                    break;
            }
        }

        private void HandleBufferErrorsChanged(object? sender, DataErrorsChangedEventArgs args)
        {
            RaiseCanExecuteChanged();
        }

        private EditBuffer EditBuffer { get; }
    }

    private class CancelCommand : Command
    {
        public CancelCommand(EditBuffer editBuffer)
        {
            EditBuffer = editBuffer;

            EditBuffer.PropertyChanged += HandleEditBufferPropertyChanged;
        }

        public override bool CanExecute(object? parameter)
        {
            return EditBuffer.HasChanges;
        }

        public override void Execute(object? parameter)
        {
            if (!CanExecute(parameter))
                return;

            EditBuffer.RevertChanges();
        }

        private void HandleEditBufferPropertyChanged(object? sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(EditBuffer.HasChanges):
                    RaiseCanExecuteChanged();
                    break;
            }
        }

        private EditBuffer EditBuffer { get; }
    }

    #endregion
}

public class EditBuffer<T> : EditBuffer
    where T : IEditBufferCloneSource<T>
{
    #region Construction

    public EditBuffer(T entity, bool isNew = false)
        : base(entity, isNew)
    {
    }

    #endregion

    #region Public

    public new T Entity => (T)base.Entity;

    public new T Source => (T)base.Source;

    #endregion

    #region Protected

    protected override void ReleaseResources()
    {
        base.ReleaseResources();
    }

    #endregion
}
