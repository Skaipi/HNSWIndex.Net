namespace HNSWIndex
{
    public class KNNResult<TLabel, TDistance>
    {
        public int Id { get; private set; }
        public TLabel Label { get; private set; }
        public TDistance Distance { get; private set; }

        internal KNNResult(int id, TLabel label, TDistance distance)
        {
            Id = id;
            Label = label;
            Distance = distance;
        }
    }
}
