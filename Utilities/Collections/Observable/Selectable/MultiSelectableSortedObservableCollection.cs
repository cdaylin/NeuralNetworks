using Daylin.Utilities.Collections.Observable.Selectable;
using Daylin.Utilities.Collections.Selection;

using System.ComponentModel;

public sealed class MultiSelectableSortedObservableCollection<T>
    : SelectableSortedObservableCollection<T> where T : class, INotifyPropertyChanged
{
    public MultiSelectableSortedObservableCollection()
        : base(new MultiSelectionState<T>())
    {
    }

    public IReadOnlyCollection<T> SelectedItems
    {
        get => SelectionState.SelectedItems;
        set => SelectionState.SetSelectedItems(value);
    }

    public void SelectItem(T item)
    {
        SelectionState.Add(item);
    }

    public void DeselectItem(T item)
    {
        SelectionState.Remove(item);
    }

    protected override void RaiseSelectionChanged()
    {
        OnPropertyChanged(new PropertyChangedEventArgs(nameof(SelectedItems)));
    }

    protected override void HandleLockReleased(object? sender, EventArgs args)
    {
        IReadOnlyCollection<T> selectedItems = SelectionState.SelectedItems;

        if (selectedItems.Count == 0)
            return;

        List<T> existingSelectedItems = [];

        foreach (T item in selectedItems)
        {
            if (Contains(item))
                existingSelectedItems.Add(item);
        }

        if (existingSelectedItems.Count != selectedItems.Count)
        {
            // One or more selected items are no longer present; update selected items.
            SelectionState.SetSelectedItems(existingSelectedItems);
        }
        else
        {
            // Selected items are still present, but view may have lost sync.
            // Force UI to rebind to valid value.
            RaiseSelectionChanged();
        }
    }

    private new MultiSelectionState<T> SelectionState => (MultiSelectionState<T>)base.SelectionState;
}