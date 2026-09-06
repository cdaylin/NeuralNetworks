namespace Daylin.Utilities.Collections.Observable.Observers;

public class CollectionChangedEventArgs<T> : EventArgs
{
    public CollectionChangedEventArgs(T item)
    {
        Item = item;
    }

    public T Item { get; }
}
