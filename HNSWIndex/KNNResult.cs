namespace HNSWIndex
{
    public class KNNResult<TItem, TDistance>
    {
        public int Id { get; private set; }
        public TItem Label { get; private set; }
        public TDistance Distance { get; private set; }

        internal KNNResult(int id, TItem label, TDistance distance) 
        { 
            Id = id;
            Label = label;
            Distance = distance;
        }
    }
}
