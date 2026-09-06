namespace Daylin.NeuralNetworks;

[TestClass]
public class IndexMapTests
{
    [TestMethod]
    public void TestIndexMap()
    {
        // this map should reverse any sequence of 5 values
        IndexMap map = new(Enumerable.Range(0, 5).Reverse());

        IEnumerable<int> original = new int[] { 1, 2, 3, 4, 5 };

        Assert.IsTrue(map.MapForward(original).SequenceEqual(original.Reverse()));
        Assert.IsTrue(map.MapBackward(original).SequenceEqual(original.Reverse()));
        Assert.IsTrue(map.MapBackward(map.MapForward(original)).SequenceEqual(original));
    }
}
