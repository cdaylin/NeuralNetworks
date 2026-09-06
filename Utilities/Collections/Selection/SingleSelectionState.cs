using Daylin.Utilities.Disposable;
using Daylin.Utilities.Observable;

namespace Daylin.Utilities.Collections.Selection;

public sealed class SingleSelectionState<T> : ObservableObject, ISelectionState<T>
    where T : class
{
    public SingleSelectionState()
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

    public T? SelectedItem
    {
        get;

        set
        {
            if (IsLocked)
                return;

            if (SetProperty(ref field, value))
                RaiseSelectionChangedEvent();
        }
    }

    public IReadOnlyCollection<T> SelectedItems => SelectedItem is null ? [] : [SelectedItem];

    private volatile int lockCount;

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
        public SelectionLock(SingleSelectionState<T> owner)
        {
            this.owner = owner;
        }

        protected override void ReleaseResources()
        {
            if (--owner.lockCount == 0)
                owner.RaiseLockReleasedEvent();
        }

        private readonly SingleSelectionState<T> owner;
    }
}