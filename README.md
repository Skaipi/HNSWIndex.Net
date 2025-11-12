# HNSWIndex
Perform KNN Query for millions of data points fast and with great accuracy. 

**HNSWIndex** is a .NET library for constructing approximate nearest-neighbor (ANN) indices based on the _Hierarchical Navigable Small World_ (HNSW) graph. This data structure provides efficient similarity searches for large, high-dimensional datasets.

## Key Features
 - **High Performance**: Implements the HNSW algorithm for fast approximate k-NN search.
 - **Flexible Distance Metric**: Pass any `Func<TVector, TVector, TDistance>` for custom distance calculation.
 - **Flexible Heuristic**: Pass heuristic function for nodes linking.
 - **Concurrency Support**: Thread safe graph building API 
 - **Configurable Parameters**: Fine-tune the indexing performance and memory trade-offs with parameters
 - **Save and Load**: Save resulting structure on file system and restore later
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
```c#
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
```c#
var index = new HNSWIndex<float[], float>(Metrics.SquaredEuclideanMetric.Compute, parameters);
```
### 3. Build the graph
```c#
var vectors = RandomVectors();
foreach (var vector in vectors)
{
	index.Add(vector);
}
```
Or multi-threaded
```c#
var vectors = RandomVectors();
Parallel.For(0, vectors.Count, i => {
    index.Add(vectors[i]);
});
```
### 4. Query the structure
```c#
var k = 5;
var results = index.KnnQuery(queryPoint, k);
```
### 5. Save and Load graph from file system
```c#
index.Serialize(pathToFile);
var index = HNSWIndex<float[], float>.Deserialize(Metrics.SquaredEuclideanMetric.Compute, pathToFile);
```
## Concurrency notes
Operations are **thread-safe per type**. You may run multiple operations of the same type in parallel on a single index instance. Mixing **different** operation types concurrently on the same index instance is not supported.
## Parameters
 - **MaxEdges** - Maximum number of outgoing edges per node. Sometimes labeld as **M**
 - **MaxCandidates** - Number of nodes resolved during insert operation. Sometimes labeled as **efConstruction**
 - **CollectionSize** - Expected number of elements that will be stored. Index is fully dynamic, however, often resizes might impact performance.
 - **DistributionRate** - Distribution rate used to promote nodes to higher levels of the graph.
 - **MinNN** - The minimal number of nodes obtained by knn search. If provided k exceeds this value, the search result will be trimmed to k. Sometimes labeled as **efSearch**.
 - **RandomSeed** - Seed for internal RNG.
 - **AllowRemovals** - Indicates if removals are allowed in the index.
# Python bindings
## Installation
```
pip install hnswindex
```
## Example usage
```py
import numpy as np
from hnswindex import Index

vectors = np.random.rand(2_000, 128)

# Create index (metric options: "sq_euclid", "cosine", "ucosine")
index = Index(dim=128, metric="sq_euclid")
index.set_collection_size(2_000)

# Batch add data
ids = index.add(vectors)

# Batch query data
# Note: distances are squared Euclidean when metric="sq_euclid".
ids, distances = index.knn_query(vectors, k=1)
```
# License
This software is licensed under the [MIT](LICENSE.TXT) license