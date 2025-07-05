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
### 🔹 Must-Know Subtopics
* ✅ **Bellman-Ford Algorithm**
* ✅ **Floyd-Warshall (All-Pairs Shortest Path)**
* ✅ **Minimum Spanning Tree: Prim’s and Kruskal’s**
* ✅ **Union-Find (Path Compression + Size/Rank)**
* ✅ **Bipartite Check**
* ✅ **Tarjan’s Algorithm (SCC, Bridges)**
* ✅ **Kosaraju’s Algorithm (SCC in Directed Graphs)**
* ✅ **Grid as Graph (Islands, Shortest Path)**

---

### 🔹 High-Yield Interview Problems

| Problem                                                                                           | Platform | Concept            |
| ------------------------------------------------------------------------------------------------- | -------- | ------------------ |
| [Number of Islands](https://leetcode.com/problems/number-of-islands)                              | Leetcode | Grid, DFS/BFS      |
| [Course Schedule](https://leetcode.com/problems/course-schedule)                                  | Leetcode | Topo Sort          |
| [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/)                               | Leetcode | Topo Sort          |
| [Clone Graph](https://leetcode.com/problems/clone-graph/)                                         | Leetcode | DFS/BFS            |
| [Network Delay Time](https://leetcode.com/problems/network-delay-time/)                           | Leetcode | Dijkstra           |
| [Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/) | Leetcode | Bellman-Ford       |
| [Find Critical Connections](https://leetcode.com/problems/critical-connections-in-a-network/)     | Leetcode | Tarjan's Algorithm |
| [Is Graph Bipartite?](https://leetcode.com/problems/is-graph-bipartite/)                          | Leetcode | Graph Coloring     |
| [Accounts Merge](https://leetcode.com/problems/accounts-merge/)                                   | Leetcode | Union-Find         |


### 🔹 Time and Space Complexity Summary

| Operation          | Time           | Space    |
| ------------------ | -------------- | -------- |
| BFS/DFS            | O(V + E)       | O(V)     |
| Dijkstra           | O(E log V)     | O(V + E) |
| Bellman-Ford       | O(V × E)       | O(V)     |
| Floyd-Warshall     | O(V³)          | O(V²)    |
| Kruskal (MST)      | O(E log E)     | O(V)     |
| Prim (MST w/ heap) | O(E log V)     | O(V)     |
| Union-Find         | O(α(N)) ≈ O(1) | O(N)     |

---

### 🔹 Code Snippets for Each Must-Know Topic

#### ✅ Bellman-Ford Algorithm

```python
def bellman_ford(n, edges, src):
    dist = [float('inf')] * n
    dist[src] = 0
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    return dist
```

---

#### ✅ Floyd-Warshall Algorithm

```python
def floyd_warshall(matrix):
    V = len(matrix)
    dist = [[matrix[i][j] for j in range(V)] for i in range(V)]
    for k in range(V):
        for i in range(V):
            for j in range(V):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist
```

---

#### ✅ Prim’s MST (Min-Heap)

```python
import heapq
def prims_mst(graph, n):
    visited = [False]*n
    min_heap = [(0, 0)]  # (weight, node)
    mst_cost = 0
    while min_heap:
        cost, u = heapq.heappop(min_heap)
        if visited[u]: continue
        visited[u] = True
        mst_cost += cost
        for v, w in graph[u]:
            if not visited[v]:
                heapq.heappush(min_heap, (w, v))
    return mst_cost
```

---

#### ✅ Kruskal’s MST (using Union-Find)

```python
def kruskal(n, edges):
    uf = UnionFind(n)
    edges.sort(key=lambda x: x[2])
    mst_cost = 0
    for u, v, w in edges:
        if uf.union(u, v):
            mst_cost += w
    return mst_cost
```

---

#### ✅ Union-Find with Path Compression & Size

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1]*n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.size[xr] < self.size[yr]:
            self.parent[xr] = yr
            self.size[yr] += self.size[xr]
        else:
            self.parent[yr] = xr
            self.size[xr] += self.size[yr]
        return True
```

---

#### ✅ Bipartite Check (BFS Coloring)

```python
def is_bipartite(graph):
    color = {}
    for node in range(len(graph)):
        if node not in color:
            queue = [node]
            color[node] = 0
            while queue:
                u = queue.pop()
                for v in graph[u]:
                    if v in color:
                        if color[v] == color[u]:
                            return False
                    else:
                        color[v] = 1 - color[u]
                        queue.append(v)
    return True
```

---

#### ✅ Tarjan’s Algorithm (Bridges)

```python
def tarjan_bridges(n, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    res, time = [], [0]
    low = [0]*n
    disc = [-1]*n

    def dfs(u, parent):
        disc[u] = low[u] = time[0]
        time[0] += 1
        for v in graph[u]:
            if disc[v] == -1:
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    res.append([u, v])
            elif v != parent:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if disc[i] == -1:
            dfs(i, -1)
    return res
```

---

#### ✅ Kosaraju’s Algorithm (SCC in Directed Graph)

```python
def kosaraju_scc(n, edges):
    from collections import defaultdict
    graph = defaultdict(list)
    rev_graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        rev_graph[v].append(u)

    visited = set()
    stack = []

    def dfs(u):
        visited.add(u)
        for v in graph[u]:
            if v not in visited:
                dfs(v)
        stack.append(u)

    for i in range(n):
        if i not in visited:
            dfs(i)

    def dfs_rev(u, comp):
        visited.add(u)
        comp.append(u)
        for v in rev_graph[u]:
            if v not in visited:
                dfs_rev(v, comp)

    visited.clear()
    scc = []
    while stack:
        u = stack.pop()
        if u not in visited:
            comp = []
            dfs_rev(u, comp)
            scc.append(comp)
    return scc
```

---

#### ✅ Graphs in Grid (Shortest Path in Binary Matrix)

```python
from collections import deque
def shortest_path_binary_matrix(grid):
    n = len(grid)
    if grid[0][0] != 0 or grid[n-1][n-1] != 0:
        return -1
    directions = [(0,1),(1,0),(1,1),(-1,-1),(-1,0),(0,-1),(-1,1),(1,-1)]
    q = deque([(0, 0, 1)])
    visited = set((0, 0))
    while q:
        r, c, d = q.popleft()
        if (r, c) == (n-1, n-1):
            return d
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc, d + 1))
    return -1
```


Let me know if you want this in **Markdown** or **PDF format**, or want a **cheat sheet for another topic like DP or Recursion**.

---

### ✅ Final Interview Readiness Checklist

* [ ] Can I implement BFS/DFS for any input form (list, matrix, object)?
* [ ] Can I detect cycles in both directed & undirected graphs?
* [ ] Can I handle shortest paths with Dijkstra/Bellman-Ford?
* [ ] Can I model a real-world problem as a graph?
* [ ] Can I implement Union-Find with optimizations?
* [ ] Am I confident solving grid-as-graph problems quickly?
* [ ] Can you write BFS/DFS for any structure (graph, grid)?
* [ ] Can you detect cycles with DFS and Union-Find?
* [ ] Can you explain and implement shortest path algorithms (Dijkstra, Bellman-Ford, Floyd-Warshall)?
* [ ] Can you apply Kruskal/Prim for MST?
* [ ] Can you identify SCCs with Tarjan or Kosaraju?
* [ ] Can you solve grid-as-graph pathfinding problems?
