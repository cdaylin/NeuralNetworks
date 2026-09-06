
using Daylin.Utilities.Extensions;

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Daylin.TestUtilities;

public static class Asserter
{
    public static ActionAssertion AssertThat(Action action)
    {
        return new ActionAssertion(action);
    }
}

public class ActionAssertion
{
    public ActionAssertion(Action action)
    {
        action.ThrowIfNull();

        Action = action;
    }

    protected Action Action { get; }

    public void Throws<T>() where T : Exception
    {
        try
        {
            Action.Invoke();
        }
        catch (Exception exception)
        {
            if (exception is T)
                return;

            throw;
        }

        Assert.Fail($"Exception of type '{typeof(T).Name}' was not thrown.");
    }
}
