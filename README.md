# HNSWIndex
**HNSWIndex** is a .NET library for constructing approximate nearest-neighbor (ANN) indices based on the _Hierarchical Navigable Small World_ (HNSW) graph. This data structure provides efficient similarity searches for large, high-dimensional datasets.

## Key Features
 - **High Performance**: Implements the HNSW algorithm for fast approximate k-NN search.
 - **Flexible Distance Metric**: Pass any `Func<TLabel, TLabel, TDistance>` for custom distance calculation.
 - **Flexible Heuristic**: Pass heuristic function for nodes linking.
 - **Concurrency Support**: Thread safe graph building API 
 - **Configurable Parameters**: Fine-tune the indexing performance and memory trade-offs with parameters
## Installation
Install via [NuGet](https://www.nuget.org/packages/HNSWIndex/):
```
dotnet add package HNSWIndex
```
Or inside your **.csproj**:
```
<PackageReference Include="HNSWIndex" Version="x.x.x" />
```

## Getting Started
### 1. Optionally configure parameters
```
var parameters = new HNSWParameters
{ 
    RandomSeed = 123,
    DistributionRate = 1.0,
    MaxEdges = 16,
    CollectionSize = 1024,
    // ... other parameters
};
```
### 2. Create empty graph structure ()
```
var index = new HNSWIndex<float[], float>(Metrics.SquaredEuclideanMetric.Compute);
```
### 3. Build the graph
```
var vectors = RandomVectors();
foreach (var vector in vectors)
{
	index.Add(vector)
}
```
Or multi-threaded
```
var vectors = RandomVectors();
Parallel.For(0, vectors.Count, i => {
    index.Add(vectors[i]);
});
```
### 4. Query the structure
```
var k = 5;
var results = index.KnnQuery(queryPoint, k);
```
