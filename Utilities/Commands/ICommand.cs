using System.Windows.Input;

namespace Daylin.Utilities.Commands;

/// <summary>
/// Command interface with strongly-typed command parameters.
/// </summary>
/// <typeparam name="T">
/// Type of command parameter.
/// </typeparam>
public interface ICommand<in T> : ICommand
{
    bool CanExecute(T? parameter);

    void Execute(T? parameter);
}
