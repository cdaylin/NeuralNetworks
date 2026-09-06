namespace Daylin.Utilities.Commands;

/// <summary>
/// Abstract implementation of <see cref="IAsyncCommand"/>.
/// </summary>
public abstract class AsyncCommand : Command, IAsyncCommand
{
    protected AsyncCommand()
    {
    }

    public abstract Task ExecuteAsync(object? parameter);
}

/// <summary>
/// Abstract implementation of <see cref="IAsyncCommand{T}"/>.
/// </summary>
/// <typeparam name="T">
/// Type of parameter that the command accepts. The type must be nullable.
/// </typeparam>
public abstract class AsyncCommand<T> : Command<T>, IAsyncCommand<T>
{
    protected AsyncCommand()
    {
    }

    public abstract Task ExecuteAsync(T? parameter);
}
