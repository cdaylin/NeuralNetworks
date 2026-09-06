using Daylin.Utilities.Collections.Selection;

namespace Daylin.Utilities.Collections.Observable.Selectable;

public abstract class SelectableObservableCollection<T> : ObservableList<T>
    where T : class
{
    protected SelectableObservableCollection(ISelectionState<T> selectionState)
    {
        SelectionState = selectionState;

        SelectionState.SelectionChanged += (_, _) => RaiseSelectionChanged();
        SelectionState.LockReleased += HandleLockReleased;
    }

    protected override void InsertItem(int index, T item)
    {
        using (SelectionState.Lock())
            base.InsertItem(index, item);
    }

    protected override void RemoveItem(int index)
    {
        using (SelectionState.Lock())
            base.RemoveItem(index);
    }

    protected override void MoveItem(int oldIndex, int newIndex)
    {
        using (SelectionState.Lock())
            base.MoveItem(oldIndex, newIndex);
    }

    protected override void ClearItems()
    {
        using (SelectionState.Lock())
            base.ClearItems();
    }

    protected abstract void RaiseSelectionChanged();

    protected abstract void HandleLockReleased(object? sender, EventArgs args);

    protected ISelectionState<T> SelectionState { get; }
}
