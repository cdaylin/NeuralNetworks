using Daylin.Utilities.Comparers;
using Daylin.Utilities.Disposable;
using Daylin.Utilities.Observable;

namespace Daylin.Utilities.Collections.Selection;

public sealed class MultiSelectionState<T> : ObservableObject, ISelectionState<T>
    where T : class
{
    public MultiSelectionState()
    {
    }

    public event EventHandler? SelectionChanged;

    public event EventHandler? LockReleased;

    public bool IsLocked => lockCount > 0;

    public IDisposable Lock()
    {
        lockCount++;

        return new SelectionLock(this);
    }

    public IReadOnlyCollection<T> SelectedItems => SelectedItemSet;

    public void SetSelectedItems(IEnumerable<T> items)
    {
        items.ThrowIfNull();

        if (IsLocked)
            return;

        SelectedItemSet.Clear();
        SelectedItemSet.UnionWith(items);

        // Note: This will raise events even when the new set of selected items is the same as the old.
        //  If this becomes a problem, do a set compare of new items to old items, and skip if equal.
        OnSelectionChanged();
    }

    public void Clear()
    {
        if (IsLocked)
            return;

        if (SelectedItemSet.Count == 0)
            return;

        SelectedItemSet.Clear();

        OnSelectionChanged();
    }

    public void Add(T item)
    {
        if (IsLocked)
            return;

        if (SelectedItemSet.Add(item))
            OnSelectionChanged();
    }

    public void Remove(T item)
    {
        if (IsLocked)
            return;

        if (SelectedItemSet.Remove(item))
            OnSelectionChanged();
    }

    private HashSet<T> SelectedItemSet { get; } = new(ReferenceEqualityComparer<T>.Instance);

    private volatile int lockCount;

    private void OnSelectionChanged()
    {
        RaisePropertyChangedEvent(nameof(SelectedItems));
        RaiseSelectionChangedEvent();
    }

    private void RaiseSelectionChangedEvent()
    {
        SelectionChanged?.Invoke(this, EventArgs.Empty);
    }

    private void RaiseLockReleasedEvent()
    {
        LockReleased?.Invoke(this, EventArgs.Empty);
    }

    private sealed class SelectionLock : DisposableObject
    {
        public SelectionLock(MultiSelectionState<T> owner)
        {
            this.owner = owner;
        }

        protected override void ReleaseResources()
        {
            if (--owner.lockCount == 0)
                owner.RaiseLockReleasedEvent();
        }

        private readonly MultiSelectionState<T> owner;
    }
}