namespace HNSWIndex
{
    internal struct DistanceCalculator<TLabel, TDistance>
    {
        private readonly Func<int, TLabel, TDistance> Distance;

        public TLabel Destination { get; }

        public DistanceCalculator(Func<int, TLabel, TDistance> distance, TLabel destination)
        {
            Distance = distance;
            Destination = destination;
        }

        public TDistance From(int source)
        {
            return Distance(source, Destination);
        }
    }
}
