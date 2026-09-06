using System.Windows.Input;

namespace Daylin.Utilities.Commands;

/// <summary>
/// Abstract implementation of <see cref="ICommand"/>.
/// </summary>
public abstract class Command : ICommand
{
    protected Command()
    {
    }

    public event EventHandler? CanExecuteChanged;

    public virtual bool CanExecute(object? parameter) => parameter is not null;

    public abstract void Execute(object? parameter);

    protected void RaiseCanExecuteChanged() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);
}

/// <summary>
/// Abstract implementation of <see cref="ICommand{T}"/>.
/// </summary>
/// <typeparam name="T">
/// Type of parameter that the command accepts. The type must be nullable.
/// </typeparam>
public abstract class Command<T> : ICommand<T>
{
    protected Command()
    {
        typeof(T).ThrowIfNotNullable();
    }

    public event EventHandler? CanExecuteChanged;

    bool ICommand.CanExecute(object? parameter) => CanExecute((T?)parameter);

    public virtual bool CanExecute(T? parameter) => parameter is not null;

    void ICommand.Execute(object? parameter) => Execute((T?)parameter);

    public abstract void Execute(T? parameter);

    protected void RaiseCanExecuteChanged() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);
}
