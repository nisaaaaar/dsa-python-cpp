# 📌 Graph – DSA Cheat Sheet (Python | Interview-Focused)

---

### 🔹 Topic Name: **Graphs**

---

### 🔹 Key Concepts

* A **graph** is a collection of nodes (vertices) and edges representing relationships.
* Graphs can be **directed/undirected**, **weighted/unweighted**, and **cyclic/acyclic**.
* Common real-world applications: networks, maps, dependency resolution, social graphs.

---

### 🔹 Common Patterns

* 🔁 **BFS / DFS Traversals**
* 🧠 **Cycle Detection** (Directed/Undirected)
* 🎓 **Topological Sorting** (DAGs)
* ⚖️ **Shortest Path** (Dijkstra, Bellman-Ford)
* 🧱 **Union-Find (Disjoint Set)**
* 🌈 **Graph Coloring** (Bipartite check)
* 📦 **Multi-source BFS** (grids, fire spread, rotten oranges)

---

### 🔹 Must-Know Subtopics

* Graph Representations (Adjacency List/Matrix)
* BFS & DFS (Recursive & Iterative)
* Cycle Detection (DFS/Union-Find)
* Topological Sort (DFS/Kahn’s Algo)
* Dijkstra’s Algorithm
* Bellman-Ford Algorithm
* Floyd-Warshall (All-Pairs Shortest Path)
* Minimum Spanning Tree (Prim's and Kruskal's)
* Union-Find with Path Compression & Rank
* Bipartite Check
* Tarjan’s Algorithm (SCC, Bridges)
* Kosaraju’s Algorithm
* Graphs in Grid (Island, Shortest Path)

---

### 🔹 High-Yield Interview Problems

| Problem                                                                 | Platform | Concept      |
| ----------------------------------------------------------------------- | -------- | ------------ |
| [Number of Islands](https://leetcode.com/problems/number-of-islands)    | Leetcode | BFS/DFS Grid |
| [Course Schedule](https://leetcode.com/problems/course-schedule)        | Leetcode | Topo Sort    |
| [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/)     | Leetcode | Topo Sort    |
| [Clone Graph](https://leetcode.com/problems/clone-graph/)               | Leetcode | DFS/BFS      |
| [Network Delay Time](https://leetcode.com/problems/network-delay-time/) | Leetcode | Dijkstra     |

---

### 🔹 Python Implementation Tips

* Use `defaultdict(list)` for adjacency list
* Use `deque` for BFS (O(1) pop from front)
* Use `heapq` for Dijkstra (priority queue)
* `visited` should be a `set` for fast lookup
* For undirected graphs, add edges both ways
* Use `enumerate()` and 2D direction arrays for grid problems

---

### 🔹 Time and Space Complexity Summary

| Operation                     | Time Complexity | Space Complexity |
| ----------------------------- | --------------- | ---------------- |
| BFS/DFS                       | O(V + E)        | O(V)             |
| Dijkstra (min-heap)           | O(E log V)      | O(V + E)         |
| Bellman-Ford                  | O(V × E)        | O(V)             |
| Floyd-Warshall                | O(V³)           | O(V²)            |
| Kruskal's (MST)               | O(E log E)      | O(V)             |
| Union-Find (w/ optimizations) | O(α(N)) ≈ O(1)  | O(N)             |

---

### 🔹 Code Snippets (For Must-Know Topics)

#### ✅ Graph Representation (Adjacency List)

```python
from collections import defaultdict
graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
```

#### ✅ DFS Traversal

```python
def dfs(node, visited, graph):
    visited.add(node)
    for nei in graph[node]:
        if nei not in visited:
            dfs(nei, visited, graph)
```

#### ✅ BFS Traversal

```python
from collections import deque

def bfs(start, graph):
    visited = set([start])
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nei in graph[node]:
            if nei not in visited:
                visited.add(nei)
                queue.append(nei)
```

#### ✅ Cycle Detection (Undirected using DFS)

```python
def has_cycle(node, parent, visited, graph):
    visited.add(node)
    for nei in graph[node]:
        if nei not in visited:
            if has_cycle(nei, node, visited, graph):
                return True
        elif nei != parent:
            return True
    return False
```

#### ✅ Topological Sort (Kahn’s Algorithm - BFS)

```python
from collections import deque, defaultdict

def topo_sort(V, edges):
    graph = defaultdict(list)
    indegree = [0] * V
    for u, v in edges:
        graph[u].append(v)
        indegree[v] += 1

    q = deque([i for i in range(V) if indegree[i] == 0])
    topo = []
    while q:
        node = q.popleft()
        topo.append(node)
        for nei in graph[node]:
            indegree[nei] -= 1
            if indegree[nei] == 0:
                q.append(nei)
    return topo
```

#### ✅ Dijkstra’s Algorithm (Min-Heap)

```python
import heapq

def dijkstra(graph, start):
    dist = {start: 0}
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        for v, w in graph[u]:
            if v not in dist or d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(heap, (dist[v], v))
    return dist
```

#### ✅ Union-Find (with Path Compression + Union by Rank)

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1
        return True
```

#### ✅ Grid as Graph (e.g., Number of Islands)

```python
def num_islands(grid):
    rows, cols = len(grid), len(grid[0])
    visited = set()

    def dfs(r, c):
        if (r < 0 or r >= rows or
            c < 0 or c >= cols or
            grid[r][c] == '0' or
            (r, c) in visited):
            return
        visited.add((r, c))
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)

    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1' and (r, c) not in visited:
                dfs(r, c)
                count += 1
    return count
```

---

### ✅ Final Interview Readiness Checklist

* [ ] Can I implement BFS/DFS for any input form (list, matrix, object)?
* [ ] Can I detect cycles in both directed & undirected graphs?
* [ ] Can I handle shortest paths with Dijkstra/Bellman-Ford?
* [ ] Can I model a real-world problem as a graph?
* [ ] Can I implement Union-Find with optimizations?
* [ ] Am I confident solving grid-as-graph problems quickly?
