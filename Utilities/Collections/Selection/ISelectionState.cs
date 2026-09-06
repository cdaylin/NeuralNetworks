namespace Daylin.Utilities.Collections.Selection;

public interface ISelectionState<T>
    where T : class
{
    event EventHandler? SelectionChanged;

    event EventHandler? LockReleased;

    IReadOnlyCollection<T> SelectedItems { get; }

    IDisposable Lock();
}