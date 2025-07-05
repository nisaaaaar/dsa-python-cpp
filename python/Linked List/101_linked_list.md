# 🧾 Linked List – DSA Interview Cheat Sheet

---

### 📌 1. Topic Overview

A **Linked List** is a linear data structure where elements (nodes) are stored in **non-contiguous** memory locations. Each node contains:

* A **data** field
* A **pointer/reference** to the next node

Types include:

* **Singly Linked List:** Each node points to the next.
* **Doubly Linked List:** Each node points to both the next and previous nodes.
* **Circular Linked List:** Last node points back to the head.

---

### 🔑 2. Key Concepts

* **Node Structure:** Basic unit containing `value` and `next`.
* **Head & Tail:** Start and (optionally tracked) end of the list.
* **Traversal:** Iterate from head until `None` (or head in circular).
* **Insertion/Deletion:**

  * At beginning, end, or a specific position.
* **Reversing a list:** Classic problem (iterative or recursive).
* **Slow & Fast Pointer Technique:** Detect cycles, find middle.
* **Cycle Detection:** Floyd’s Tortoise and Hare Algorithm.
* **Dummy Node Usage:** For simplifying insert/delete edge cases.
* **Merge Two Sorted Lists:** Common merge pattern.
* **Linked List as Stack/Queue:** Efficient head/tail operations.

---

### 📚 3. Common Interview Problems

| Problem                                                                                          | Description                               |
| ------------------------------------------------------------------------------------------------ | ----------------------------------------- |
| [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)                        | Reverse a singly linked list              |
| [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)                            | Detect if a cycle exists                  |
| [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)                  | Merge two sorted linked lists             |
| [Remove N-th Node from End](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)     | Remove the nth node from the end          |
| [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)                  | Check if list is a palindrome             |
| [Intersection of Two Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/)     | Find the node where two lists intersect   |
| [Flatten Multilevel DLL](https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/) | Flatten a nested doubly linked list       |
| [Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)    | Clone complex linked list                 |
| [Add Two Numbers](https://leetcode.com/problems/add-two-numbers/)                                | Add numbers represented by linked lists   |
| [Sort List](https://leetcode.com/problems/sort-list/)                                            | Merge sort implementation for linked list |

---

### 🧩 4. Patterns and Techniques

* **Two Pointer Pattern (Slow & Fast):**

  * Find middle, detect cycle, palindrome check
* **Dummy Node Pattern:**

  * Simplifies head manipulations in add/remove
* **Recursive Patterns:**

  * Reverse, clone list, divide-and-conquer (merge sort)
* **Stack-based Approach:**

  * Used in palindrome checks, backtracking
* **In-place Reversal / Swapping:**

  * Reverse list or pairs/groups of nodes

---

### 📊 5. Time & Space Complexities

| Operation        | Singly LL   | Doubly LL   | Notes                |
| ---------------- | ----------- | ----------- | -------------------- |
| Insert at head   | O(1)        | O(1)        | Constant time        |
| Insert at tail   | O(n)\*/O(1) | O(n)\*/O(1) | O(1) if tail tracked |
| Delete from head | O(1)        | O(1)        | -                    |
| Delete from tail | O(n)        | O(1)        | O(1) in DLL          |
| Search by value  | O(n)        | O(n)        | Linear scan          |
| Reverse list     | O(n)        | O(n)        | In-place possible    |

---

### ✅ 6. Best Practices and Tips

**Dos:**

* Always handle edge cases (`null`, one-node list)
* Use **dummy nodes** for cleaner logic
* Use **slow/fast pointers** for length-based problems
* Know both **iterative and recursive** versions of reversal

**Don'ts:**

* Don’t forget to update `.next` or `.prev` when modifying links
* Don’t assume list length unless you've computed it
* Avoid extra space unless necessary (aim for in-place)

---

### 🧪 7. Code Snippets (Python)

#### 1. Node Definition (Singly)

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

#### 2. Reverse a Linked List

```python
def reverseList(head):
    prev = None
    curr = head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev
```

#### 3. Detect Cycle (Floyd's Algo)

```python
def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

#### 4. Merge Two Sorted Lists

```python
def mergeTwoLists(l1, l2):
    dummy = curr = ListNode()
    while l1 and l2:
        if l1.val < l2.val:
            curr.next, l1 = l1, l1.next
        else:
            curr.next, l2 = l2, l2.next
        curr = curr.next
    curr.next = l1 or l2
    return dummy.next
```

---

### 🧠 8. Real-World Applications

* **Memory-efficient data structures** where array resizing is expensive
* **Dynamic data storage** in OS (like process control blocks)
* **Undo functionality** in editors (doubly linked lists)
* **Implementing stacks, queues, hashmaps (chaining)**
* **Graph adjacency list** representation

---

### 🖼️ 9. Visuals

#### Singly Linked List:

```
[1] -> [2] -> [3] -> None
```

#### Reversed List:

```
None <- [1] <- [2] <- [3]
```

#### Fast & Slow Pointer Cycle Detection:

```
          fast
           ↓
head -> [1] -> [2] -> [3] -
                   ↑      ↓
                  [5] <- [4]
```

---