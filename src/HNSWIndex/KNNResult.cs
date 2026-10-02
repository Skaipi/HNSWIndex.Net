namespace HNSWIndex
{
    public class KNNResult<TVector, TDistance>
    {
        public int Id { get; private set; }
        public TVector Label { get; private set; }
        public TDistance Distance { get; private set; }

        internal KNNResult(int id, TVector label, TDistance distance)
        {
            Id = id;
            Label = label;
            Distance = distance;
        }
    }
}
