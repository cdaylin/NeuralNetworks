namespace Daylin.Utilities.Validation;

public abstract class DataValidator<TEntity, TError> : IDataValidator<TError>
{
    protected DataValidator()
    {
    }

    public abstract IEnumerable<TError> GetErrors(TEntity entity, string? propertyName);

    IEnumerable<TError> IDataValidator<TError>.GetErrors(object dataEntity, string? propertyName)
    {
        dataEntity.ThrowIfNotAssignableTo<TEntity>();

        return GetErrors((TEntity)dataEntity, propertyName);
    }
}
