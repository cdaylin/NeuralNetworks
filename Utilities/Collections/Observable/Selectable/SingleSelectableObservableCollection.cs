using Daylin.Utilities.Collections.Selection;

using System.ComponentModel;

namespace Daylin.Utilities.Collections.Observable.Selectable;

public sealed class SingleSelectableObservableCollection<T>
    : SelectableObservableCollection<T>, ISingleSelectableObservableCollection<T>
    where T : class
{
    public SingleSelectableObservableCollection()
        : base(new SingleSelectionState<T>())
    {
    }

    public T? SelectedItem
    {
        get => SelectionState.SelectedItem;
        set => SelectionState.SelectedItem = value;
    }

    protected override void RaiseSelectionChanged()
    {
        OnPropertyChanged(new PropertyChangedEventArgs(nameof(SelectedItem)));
    }

    protected override void HandleLockReleased(object? sender, EventArgs args)
    {
        T? selectedItem = SelectionState.SelectedItem;

        if (selectedItem is null)
            return;

        if (!Contains(selectedItem))
            // Selected item is no longer present; clear it.
            SelectionState.SelectedItem = null;
        else
            // Item is still selected and present, but view may have lost sync.
            // Force UI to rebind to valid value.
            RaiseSelectionChanged();
    }

    private new SingleSelectionState<T> SelectionState => (SingleSelectionState<T>)base.SelectionState;
}
