using System.Windows.Input;

namespace Daylin.Utilities.Commands;

public interface IAsyncCommand : ICommand
{
    /// <summary>
    /// Executes the command asynchronously.
    /// </summary>
    /// <remarks>
    /// This method should perform the same operation as <see cref="ICommand.Execute(object?)"/>, but in such a
    /// way that the operation can be awaited.
    /// </remarks>
    /// <param name="parameter">
    /// The command parameter.
    /// </param>
    /// <returns>
    /// Returns a task that represents this asynchronous operation.
    /// </returns>
    Task ExecuteAsync(object? parameter);
}

/// <summary>
/// A command that can execute asynchronously.
/// </summary>
/// <typeparam name="T">
/// Type of parameter that the command accepts.
/// </typeparam>
public interface IAsyncCommand<in T> : ICommand<T>
{
    /// <summary>
    /// Executes the command asynchronously.
    /// </summary>
    /// <remarks>
    /// This method should perform the same operation as <see cref="ICommand.Execute(object?)"/>, but in such a
    /// way that the operation can be awaited.
    /// </remarks>
    /// <param name="parameter">
    /// The command parameter.
    /// </param>
    /// <returns>
    /// Returns a task that represents this asynchronous operation.
    /// </returns>
    Task ExecuteAsync(T? parameter);
}
