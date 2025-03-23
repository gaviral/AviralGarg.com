---
layout: main
title: Sorting Algorithms
---

<link rel="stylesheet" type="text/css" href="../../style.css">

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
      inlineMath: [['$','$']]
    }
  });
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script> 

# Sorting


## 1. Introduction to Sorting Problem
- **Sorting Problem**: Reorder a sequence of $ n $ elements into non-decreasing order.
- **Extensions**: Applicable to numbers, characters, strings, and comparable complex objects.

### 1.1 Importance of Sorting

  - **Efficiency**: Sorted data allows binary search, reducing search time compared to linear search on unsorted data.
  - **Duplicate Detection**: Sorting arranges identical elements consecutively. That facilitates single-pass algorithms to identify or remove duplicates.
  - **Merging Sorted Files**: Efficiently combines multiple sorted lists into a single sorted list. That allows aggregating data from multiple sources with sorted inputs.
  - **Order Statistics**
    - **Median**: Middle element in a sorted sequence.
    - **Kth Smallest/Largest**: Direct access in a sorted array.
  - **Top K Elements**: Easily retrievable from the first or last $ k $ positions in a sorted list.
  - **User Interface Applications**
    - **Truncated Displays**:
      - **Examples**: Email inboxes, search results, news feeds.
      - **Purpose**: Present top relevant items from a sorted dataset.

### 1.2 Benefits of Studying Sorting Algorithms

#### Learning Design Strategies

- **Divide and Conquer**:
  - Break problem into smaller subproblems.
  - Recursively solve and combine results.
- **Decrease and Conquer**:
  - Reduce problem size incrementally.
  - Solve smaller instances to build up to the solution.
- **Applicability**: Strategies extend beyond sorting to various algorithmic challenges.

#### Algorithm Analysis

- **Performance Evaluation**:
  - Time Complexity: Assess efficiency (e.g., O(n log n)).
  - Space Complexity: Determine memory usage.
- **Comparison**: Enable selection of optimal algorithms based on problem constraints.

### 1.3 Approach to Learning Sorting Algorithms

- **Strategy Focus**: Emphasize understanding design principles over memorizing specific algorithms.
- **Algorithm Derivation**: Recognize how design strategies lead to different sorting methods.

### 1.4 Summary

- **Essential Skill**: Sorting underpins efficient data processing and various algorithmic applications.
- **Strategic Mastery**: Understanding design strategies enables tackling diverse and complex problems.

---

## 2. Design Strategy: Brute Force - Selection Sort

- **Brute Force**: Simplest and most straightforward approach based directly on the problem statement.

### 2.1 Selection Sort Algorithm

- **Concept**: Repeatedly select the smallest remaining element and place it in its correct position in the array.
- **Process**:
  - For each index $ i $ from $ 0 $ to $ n - 1 $:
    - Find the minimum element in the subarray from $ A[i] $ to $ A[n-1] $.
    - Swap this minimum element with $ A[i] $.

```python
def selection_sort(A):
    n = len(A)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if A[j] < A[min_index]:
                min_index = j
        # Swap the found minimum element with the first element
        A[i], A[min_index] = A[min_index], A[i]
    return A
```

- **Explanation**:
  - **Outer Loop** ($ i $ from $ 0 $ to $ n - 1 $):
    - Sets the current position for the next smallest element.
  - **Inner Loop** ($ j $ from $ i + 1 $ to $ n - 1 $):
    - Finds the minimum element in the unsorted subarray.
  - **Swap Operation**:
    - Exchanges $ A[i] $ with $ A[min\_index] $ to place the smallest element at position $ i $.

### 2.2 Correctness Argument

- **Invariant**: After the $ i $-th iteration, the first $ i $ elements are sorted and are the smallest elements in the array.
- **Base Case**: Before any iteration, $ i = 0 $, and no elements are sorted.
- **Inductive Step**:
  - The algorithm selects the smallest element from the unsorted subarray and places it at position $ i $.
  - Ensures that the sorted portion remains sorted after each iteration.
- **Termination**:
  - After $ n $ iterations, the entire array is sorted in non-decreasing order.

### 2.3 Performance Analysis

#### Time Complexity

- **Total Comparisons**:
  - Outer loop runs $ n $ times.
  - Inner loop runs $ n - i - 1 $ times for each $ i $.
  - **Total**: $ O(n^2) $ comparisons.
- **Worst, Average, and Best Case**:
  - All cases have $ O(n^2) $ time complexity since comparisons are independent of the initial order.

#### Space Complexity

- **In-Place Sorting**:
  - Uses constant extra space $ O(1) $.
  - Only a few variables for indices and temporary storage during swaps.

### 2.4 Importance

- **Understanding Basic Sorting Algorithms**:
  - Fundamental for algorithmic problem-solving.
  - Demonstrates grasp of algorithm design and analysis.
- **Algorithm Analysis Skills**:
  - Ability to argue correctness and efficiency.
  - Critical for optimizing solutions in coding interviews.
- **Optimization Awareness**:
  - Recognize when a simple algorithm may not be efficient for large inputs.
  - Prepared to discuss more efficient alternatives (e.g., Quick Sort, Merge Sort).

### 2.5 General Notes on Algorithm Analysis

- **Resource Considerations**:
  - **Time Complexity**: Measures the algorithm's efficiency as input size grows.
  - **Space Complexity**: Evaluates the additional memory required.
- **Focus on Time vs. Space**:
  - Time is often prioritized due to processor speed limitations.
  - Space can sometimes be mitigated with additional memory resources.
- **Performance Metrics**:
  - Important to understand Big O notation and its implications on scalability.

### 2.6 Summary

- **Selection Sort**:
  - Simple, intuitive sorting algorithm.
  - Not efficient for large datasets due to $ O(n^2) $ time complexity.
- **Key Takeaways for Interviews**:
  - Ability to implement basic algorithms from scratch.
  - Skills in analyzing and articulating algorithm efficiency.
  - Foundation for understanding more advanced sorting algorithms.

---

## 3.Basics of Asymptotic Analysis

### 3.1 Quantifying Algorithm Running Time

- **System-Dependent Factors**:
  - Hardware capabilities (processor speed, architecture).
  - Operating system load and multitasking.
  - Programming language and compiler optimizations.

- **System-Independent Factors**:
  - **Input Size ($ n $)**:
    - Primary determinant of running time.
    - Larger inputs generally require more computation.

### 3.2 Defining Running Time as a Function of Input Size

- **Notation**: $ T(n) $ represents the running time as a function of input size $ n $.
- **Approach**:
  - Count the number of **basic operations** executed for input size $ n $.
  - Basic operations execute in a constant amount of time.

### 3.3 Basic Operations

- **Definition**: Operations with a fixed execution time, regardless of input size.
- **Examples**:
  - Variable assignments (e.g., $ x = y $).
  - Arithmetic operations (e.g., $ z = x + y $).
  - Comparisons (e.g., $ x < y $).
  - Swapping elements in an array.
- **Constants**:
  - Each basic operation has an associated time $ c_i $ (system-dependent constant).

### 3.4 Analyzing Selection Sort

#### Algorithm Overview

- **Process**:
  1. For each index $ i $ from $ 0 $ to $ n - 1 $:
     - Set $ \text{min\_index} = i $.
     - For each index $ j $ from $ i + 1 $ to $ n - 1 $:
       - If $ A[j] < A[\text{min\_index}] $, update $ \text{min\_index} = j $.
     - Swap $ A[i] $ with $ A[\text{min\_index}] $.

#### Counting Operations

- **Outer Loop** ($ n $ iterations):
  - Variable assignments and swaps:
    - Executed $ n $ times.
    - Total cost: $ (c_1 + c_2) \times n $.

- **Inner Loop**:
  - For each $ i $, inner loop runs $ n - i - 1 $ times.
  - **Total Comparisons**:
    - Sum from $ i = 0 $ to $ n - 2 $ of $ n - i - 1 $:
      - $ \sum_{i=0}^{n-2} (n - i - 1) = \frac{n(n - 1)}{2} $.
    - Total cost: $ c_3 \times \frac{n(n - 1)}{2} $.

#### Total Running Time

- **Expression**:
  - $ T(n) = a n^2 + b n + c $
    - $ a $: Aggregated constant from quadratic term.
    - $ b $: Aggregated constant from linear terms.
    - $ c $: Constant operations outside loops.
- **Dominant Term**: $ a n^2 $ (quadratic growth).

### 3.5 Asymptotic Analysis

#### Dominant and Lower-Order Terms

- **Dominant Term**:
  - Highest-degree term in $ T(n) $ (e.g., $ n^2 $).
  - Major contributor to running time for large $ n $.

- **Lower-Order Terms**:
  - Terms with lower degrees (e.g., $ n $, constants).
  - Contribution becomes negligible as $ n $ grows large.

#### Simplifying Running Time

- **Approximation for Large $ n $**:
  - $ T(n) \approx a n^2 $.
  - Ignore lower-order terms and constants.

- **Asymptotic Notation**:
  - **Theta Notation ($ \Theta $)**:
    - $ T(n) = \Theta(n^2) $ denotes quadratic growth.
    - Focuses on the dominant term's order of magnitude.

#### Implications of $ \Theta(n^2) $

- **Growth Rate**:
  - Doubling $ n $ quadruples $ T(n) $.
  - Running time increases rapidly with input size.

- **Algorithm Efficiency**:
  - Less efficient for large datasets compared to algorithms with lower time complexities (e.g., $ \Theta(n \log n) $).

### 3.6 Importance in Algorithm Analysis

- **System-Independent Evaluation**:
  - Abstracts away hardware and implementation specifics.
  - Enables comparison of algorithms based on fundamental efficiency.

- **Focus on Growth Rates**:
  - Critical for understanding scalability.
  - Helps predict performance on large inputs.

### 3.7 Practical Application in Interviews

- **Analyzing Algorithms**:
  - Derive $ T(n) $ expressions by counting operations.
  - Identify and focus on the dominant term.

- **Optimizing Solutions**:
  - Recognize inefficient algorithms ($ \Theta(n^2) $) for large $ n $.
  - Propose more efficient alternatives when necessary.

- **Developing Intuition**:
  - Quickly estimate time complexities.
  - Make informed decisions under time constraints.

---

## 4. Brute Force: Bubble Sort

### Introduction

- **Modification of Selection Sort**:
  - Instead of scanning from left to right to find the minimum element, scan from right to left.
  - **Objective**: Bubble up the minimum element to its correct position through repeated exchanges.

### Bubble Sort Algorithm

- **Concept**:
  - Compare adjacent elements and swap them if they are out of order.
  - Repeatedly pass through the array to move the next smallest element to its correct position.
- **Visualization**:
  - Treat the array vertically with the right end at the bottom.
  - The smallest (lightest) elements "bubble up" to the top (left end) of the array.

### Algorithm Steps

1. **For each index** $i$ from $0$ to $n - 1$:
   - Initialize a **j pointer** starting from index $n - 1$ and moving down to $i + 1$.
2. **For each position** of the j pointer:
   - **Compare** $A[\text{j} - 1]$ and $A[\text{j}]$.
   - **Swap** if $A[\text{j} - 1] > A[\text{j}]$.
3. **Result**:
   - After each pass, the next smallest element is placed at index $i$.
   - The sorted region grows from the left.

### Pseudocode

```python
def bubble_sort(A):
    n = len(A)
    for i in range(n):
        for j in range(n - 1, i, -1):
            if A[j - 1] > A[j]:
                # Swap adjacent elements
                A[j - 1], A[j] = A[j], A[j - 1]
    return A
```

- **Variables**:
  - **$n$**: Length of the array.
  - **$i$**: Index for each pass through the array.
  - **$\text{j}$**: Pointer moving from right to left.

### Correctness Argument

- **Invariant**:
  - After the $i$-th pass, the first $i$ elements are the smallest elements in sorted order.
- **Mechanism**:
  - The smallest element in the unsorted portion bubbles up to index $i$ through swaps.
- **Conclusion**:
  - Repeating this process $n$ times sorts the entire array.

### Time Complexity Analysis

- **Total Comparisons**:
  - The inner loop runs $n - i - 1$ times for each $i$.
  - **Total**: $\sum_{i=0}^{n-2} (n - i - 1) = \frac{n(n - 1)}{2}$ comparisons.
- **Total Swaps**:
  - Depends on the initial order; in the worst case, similar to the number of comparisons.
- **Time Complexity**:
  - **Worst-case**: $O(n^2)$.
  - **Asymptotic Notation**: $T(n) = \Theta(n^2)$.

### Comparison with Selection Sort

- **Similarities**:
  - Both have a time complexity of $\Theta(n^2)$.
  - Both are simple to understand and implement.
- **Differences**:
  - **Selection Sort**:
    - Typically performs fewer swaps (one swap per outer loop iteration).
    - Potentially more efficient in practice due to reduced swap operations.
  - **Bubble Sort**:
    - May perform many swaps per pass, leading to higher constant factors in time complexity.
    - Generally slower in practice compared to Selection Sort.

### Practical Considerations

- **Inefficiency of Bubble Sort**:
  - High number of swap operations makes it impractical for large datasets.
  - Often considered one of the least efficient sorting algorithms.
- **Educational Value**:
  - Illustrates the difference between a correct algorithm and an efficient one.
  - Serves as an example in algorithm analysis and understanding time complexity.

### Notable Remarks

- **Quote by Jim Gray (ACM Turing Award, 1998)**:
  - *"Bubble sort crisply shows the difference between a correct algorithm and a good algorithm."*
- **Historical Anecdote**:
  - During a presidential campaign visit to Google in 2007, Senator Barack Obama was asked about sorting algorithms.
    - He humorously noted that *"I think the bubble sort would be the wrong way to go."*

### Summary

- **Bubble Sort**:
  - Simple but inefficient sorting algorithm with $\Theta(n^2)$ time complexity.
  - Not suitable for large datasets due to poor performance.
- **Interview Insight**:
  - Understanding why Bubble Sort is inefficient helps in choosing better algorithms.
  - Demonstrates the importance of algorithm optimization and analysis in software development roles.

---


## 5. Decrease and Conquer Strategy and Insertion Sort

### Algorithm Design Strategies

<!-- below is HTML CODE TO MAKE A TABLE - demonstrating 2 kinds of merge cells: vertically and horizontally -->

<table>
<tr>
  <th>#</th>
  <th>Design Strategy</th>
  <th>Description</th>
  <th>Pros/Cons</th>
  <th>Examples</th>
</tr>
  <tr>
    <td>1.</td>
    <td><b>Brute Force</b></td>
    <td>- Directly based on the problem statement.</td>
    <td>✅ Straightforward<br>❌ May vary between individuals due to subjectivity.</td>
    <td>Selection Sort, Bubble Sort</td>
  </tr>
  <tr>
    <td>2.</td>
    <td><b>Decrease and Conquer</b></td>
    <td>- Decrease the problem size ($n$) to size $n-1$.
    </td>
    <td>✅: Systematic approach to problem-solving.<br>✅: Facilitates recursive thinking.</td>
    <td></td>
  </tr>
  <tr>
    <td>2.1</td>
    <td>TOP-DOWN</td>
    <td>- Do upfront work to nibble away one (or few) element(s): $n \rightarrow n-1$
      <br>
      <br><b>MAIN IDEA</b>:
      <br>1. I pretend to be a Lazy Manager.
      <br>2. I nibble away at the problem to make it smaller.
      <br>3. I do <b>upfront work</b> of adding the nibble to the solution.
      <br>4. I hand over <b>my solution</b> and the <b>remaining problem</b> to my subordinate. (They do the same.)
      <br>
      <br><b>CODING TIP</b>: Write code from the perspective of an <b>arbitrary</b> lazy manager to sort the array:
      <br>..<b>BEFORE</b>: The managers above me would have ensured that all previous elements are sorted. (because they are lazy managers too!)
      <br>..<b>MY JOB</b>: I just extend the sorted region by one more element.
      <br>..<b>AFTER</b>: I hand over the remaining work to my subordinate. (They do the same.)
    </td>
    <td></td>
    <td>Selection Sort, Bubble Sort</td>
  </tr>
  <tr>
    <td>2.2</td>
    <td>BOTTOM-UP</td>
    <td>- Work at the end to extend the $(n-1)^{st}$ solution: $n-1 \rightarrow n$
      <br>
      <br><b>MAIN IDEA</b>:
      <br>1. I pretend to be a Lazy Manager.
      <br>2. I nibble away at the problem to make it smaller. I keep the nibble for work to be done <b><i>later</i></b>.
      <br>3. I hand over the <b>remaining problem</b> to my subordinate. (They do the same.)
      <br>4. The subordinate returns their <b>solution</b> to me.
      <br>5. I do the (<b><i>later</i></b>) work of adding my nibble to the solution.
      <br>6. I hand over my <b>solution</b> to my manager.
    </td>
    <td></td>
    <td>Insertion Sort</td>
  </tr>
</table>

<!--  We now want to write the code for doing insertion sort on an integer array A. Now if you are given the entire array A to sort, you are going to assume that your subordinate has already sorted a slightly smaller prefix of the original array, and you are going to try and extend that sorted region by one more element. Your subordinate in turn will do the same thing. Assume that their subordinate has already solved slightly smaller prefix, and they will try to extend that by one more element. So when we look at an arbitrary manager in this imaginary corporate hierarchy, an arbitrary manager, let's say this red manager, is going to be given some prefix, let's say up to index i of the original array, and they are asked to sort this prefix denoted by this red rectangle. And to sort this prefix, this red manager will depend on a subordinate blue manager who is given this blue rectangle to sort. This blue rectangle extends up to index i-1. So assuming that the blue rectangle is already sorted, the red manager is going to take the rightmost element of their subproblem, which is the number A of i, and try to insert it into its appropriate position within the blue rectangle. And thereby extend the sorted region by one more element. So we will assume that this number A of i is copied into a temporary variable. And so when we write that in code, it's going to look like this. So we will see that the temp is assigned the value of a of i. And now we can start looking at all the values in the blue rectangle from right to left. And as long as we are seeing numbers that are bigger than what is in the temporary variable, we know that all of them have to be right shifted. So we will keep right shifting each number by one position. As long as we see that number as having a value that's bigger than the temporary variable, we so we will do this right to left scan. So we will do this right to left scan by using an index that we will again call the red index. And this is initialized to index i-1. And as long as the value that is pointing to, let me call that as a of red. As long as a of red is bigger than the value in the temporary variable, we are going to keep right shifting a of red. So we will say that we will move a of red to the index immediately to its right to its right. So a of red to its right. So a of red plus 1 will be assigned a of red to the right to the index. is bigger than the value in the temporary variable, we are going to keep right shifting A of red. So we'll say that we will move A of red to the index immediately to its right. So A of red plus 1 will be assigned the value of A of red. And having done this, we will keep moving the red index further one more step to the left. Now, there will come a point where the value in the red index might end up being less than or equal to temp. And as soon as I see such a spot, I know that the temp variable has to be copied into the place right next to it. Now there would be an empty spot right next to it because when we were looking at this number, whatever was in this place, that number must have been bigger than temp because of which it must have been right shifted, leaving an empty spot there. So now that we have found the right spot for temp, we have to reinsert the temp value into A of red plus 1. So now our red manager has managed to extend the length of the sorted region by one more element. And so if each manager did this in a bottom up way, then the entire array should be sorted. So we can have a manager index i varying from 0 to n minus 1 as before. And everything here, all the work that's done by an arbitrary manager is going to be repeated by each of the managers in a bottom up order. And at the end, we will return the sorted version of the original array A. Now there is one subtle scenario that we should consider here, which is that as we are doing this right to left scan, it is possible that every single number within the blue rectangle might be larger than temp. So every single number would get right shifted. And the red index would jump off the left end of the array and its value might become minus 1. Now minus 1 is not a legal index. So we cannot allow the value of red to dip below 0. So we should add an additional constraint that our red index that our red index needs to be greater than or equal to 0. So as long as the red index is pointing to a valid number within the boundary of this blue rectangle, and that number happens to be bigger than temp, we keep right shifting that number. So when we exit the while loop, it will either be because we saw a number within the blue rectangle that was smaller than or equal to temp, or our red index jumped off the left end and became minus 1. So in that scenario, we want to insert this temp value at the very beginning of the array because it's smaller than all the elements. So all the elements have been shifted on the right side. And so an empty spot is now available in the left most position. And since red is minus 1, we exit this while loop. And our code here is going to try to copy the temp value into index minus 1 plus 1, which is index 0. And that is in fact where we want to the temp value to go. So this piece of code will continue to work correctly, even if the temp value that we are trying to insert is smaller than all the elements within the blue rectangle. So let's do a dry run of this code on a simple example with six elements in an integer array. The execution of this code is going to start from the very bottom of the managerial hierarchy. So let me show the bottom most node here. And this bottom most node is going to be given the responsibility of sorting the smallest possible prefix, let's say a one element array. Because i is 0. So that means the index of relevance to the bottom most manager or lowly developer is the index 0. So we will copy this value 8 into the index value 8 into some temporary variable. So temp is going to become equal to 8. And then we are going to have our red index looking left and starting from index 0, 0 minus 1 which is minus 1. Now there are no numbers on the left. And so we are not going to enter into this while loop since red is minus 1. So we will come here and we will copy back the value from the temp variable into a of minus 1 plus 1 which is a going to enter into this while loop since red is minus 1 so we will come here and we will copy back the value from the temp variable into a of minus 1 plus 1 which is a of 0 so this value 8 got copied out into temp and now it gets copied back into the same spot now that the sub problem of size 1 is solved the next manager in sequence is going to be given a sub problem of size 2 now the subordinate of this manager has already sorted this one element prefix and so in order to sort this slightly larger sub array the index of relevance for this manager is index 1 so this manager will take this value 2 and copy it into the temporary variable so the temporary variable gets a value of 2 and then we have the red index that's going to scan everything to its left so it's going to scan this one element array and since 8 is bigger than 2 we know that 2 needs to go on the left side of 8 so we're going to go into the body of this while loop and we're going to right shift the 8 into the position to its right so we'll end up overwriting the 2 with the 8 and having right shifted the 8 our red index will now jump off the left end and it will become minus 1 so once it becomes minus 1 we are going to skip the body of this while loop and come here and copy back the temp variable into index 0 so this 2 is going to be copied back into index 0 and so at index 0 we'll have the 2 so now our 2 element prefix is sorted we will next move to the prefix of length 3 the index of relevance for the blue manager is index 2 so we are going to take the value 4 and copy it into the temporary variable so instead of this 2 we'll have a 4 and then we are going to have our red index scan the numbers on the left side the first number we look at is 8 8 is more than 4 and so we need to right shift the 8 so we will erase the 4 and overwrite it with the 8 next our red index is going to move to the 2 and 2 happens to be less than 4 so that means we have reached the spot to the right of which 4 needs to be copied so we are going to take this temporary value and copy it back into index 1 so we erase the 8 and we copy the 4 4 into index 1 and that makes our 3 element subarray also sorted next we are going to look at the 4 element subarray so the red manager is focused on index 3 and what he or she will do is to copy the value 9 into the temporary variable and then our red index is going to scan all the numbers on the left side first we look at the 8 since 8 is less than 9 we immediately reach the spot to the right of which we need to copy the temp value and so when we exit remember we always copy the temp value into the location the right of where we stop and so this 9 is going to get copied back into the same spot and so now our 4 element subarray has been sorted the next manager in sequence let's say is going to be a white manager who is given a 5 element subarray to sort ending at index 4 so the index of relevance here is 4 that's the only number that this white manager will focus on that number is 3 so the number 3 is going to be copied into the temporary variable and so temp is going to get a value of 3 and then our red index is going to scan all the numbers from index 3 onwards to the left so 9 happens to be bigger than 3 so that means we are going to right shift the 9 we are going to right shift it by by copying its value into index 4 so this becomes 9 and then we move to the next number so now the red index points at 8 8 is also more than 3 so we are going to right shift the 8 into the cell on the right side then we continue on we look at the next number it's a 4 4 is bigger than 3 so we are going to right shift the 4 and copy it into the cell on the right side and then our red index reaches this number 2 2 happens to be less than 3 so this is the point where we stop because we know that the temp value needs to sit on the immediate right side of 2 so we are going to copy the temp value into index 1 and once we do that that our five element subarray up to index 4 has been sorted finally the last manager the yellow manager who was given the entire array to sort is going to step in and the next number 2 is going to step in and we will take the last value which is 6 and copy it into the temp variable and then scan all the numbers from right to left starting from index 4 the next number 2 is going to get the length of index 4 and so we will need to right shift the 9 because we know 6 will definitely go on the left side of 9 so when we right shift it we are going to get a 9 in index 5 next our red index will look at 8 8 is also more than 6 so we are going to right shift the 8 then our red index moves further to the left we look at this 4 4 happens to be less than 6 and so we know that this 6 needs to sit on the right next to 4 on the right side so we're going to copy this value 6 into index 3 and having done that you can see that the entire array is now sorted now let's analyze this algorithm and try to figure out its time complexity so the total amount of time spent in sorting an array of length n is going to be the sum of the time used up by each manager in doing his or her own piece of work because all of them are collectively sorting the entire array so if you can figure out how much time is used up by an arbitrary manager in this chain an arbitrary manager who is going to sort some prefix of the array ending at some arbitrary index i if you can figure out this time what we need to do is to sum up this amount of work for each manager for i varying from 0 to n minus 1 and that would be the overall running time for sorting the array the time spent by each manager here corresponds to the time required to execute these lines of code within the body of the for loop and that's the work that we are summing up for i varying from 0 to n minus 1 now if you look at the instructions within the body of the for loop these three lines all are elementary operations so they're going to contribute a constant amount to the overall running time it's this while loop that is going to take potentially more time and so it's the running time of this while loop that's going to dominate the running time of this entire block because we will always be able to ignore the constant terms if we have a higher order term here so how many times are we going to execute or go inside the body of the while loop every time we go inside the body of the while loop we do a right shift operation and we move the red index one place to the left so these two also take up a constant amount of time so constant amount of work is done every time we go inside the while loop but this while loop might be executed several times how many times is it going to be executed well roughly as many times as the number of values that we see that are more than temp so all the values that we see that are more than temp are going to be right shifted so that's essentially what we are doing here so how many right shift operations will we need to do that is the number of times the body of the while loop will be executed so I can write the time spent by manager i as the number of right shift operations times some constant which is the amount of time needed to run these two instructions as well as maybe check the while loop condition plus some other constant c2 which captures the running time of these three elementary operations and because it's this term that could be significantly high the overall cost here is going to be dominated by the first term now how high could the number of right shift operations be we have seen in the example that we went through that sometimes we might find that everything on the left side within the blue rectangle might have to be right shifted just because the value that we are trying to insert is smaller than all of the values on the left so in that case we will have to right shift all i of these numbers that appear on the left and so the number of right shift operations could be as high as i but on the other hand we have also seen scenarios where we don't have to do any right shift operations at all right at the beginning we see that the value of the right shift operations are all right at the right at the beginning we see that the value of the right shift operation might might already be bigger than the value memory we have to insert is already in its right place in which case we won't need to do anygeschraham's ув any right shift operations at all. So the number of right shift ops can be as low as zero. So it can be as low as zero and it can be as high as I. And in general for different managers in this chain the number of right shift ops can be different. For some managers it could be I, for some other managers it could be as low as zero. For some other managers it might be some intermediate value between zero and I because maybe some of the elements are right shifted but not all. So this variability in the number of right shift operations at each level causes a variability in the overall running time as well. For example if every single manager had to right shift all the numbers appearing on the left side. For example if you imagine the original array as being reverse sorted. If it's reverse sorted then no matter what number you pick all the numbers on the left side are going to be larger than it. And so when we do this right shifting we'll end up right shifting every number on the left in order to find the right spot for the next number. And so the running time in that case is going to be particularly high because this term is going to be as high as it can get. On the other hand at the other extreme if our original array was completely sorted then when any manager takes up his or her job of figuring out where the ith number should go they're going to find that the ith number is already in its correct place because the the left neighbor is already smaller than it. And so the right spot for the ith number is where it already belongs. So in that case the number of right shift operations would be as low as zero. And it would be zero for all the managers. And so the overall value of t would be particularly low. And if our original array was neither sorted nor reverse sorted but maybe some of the elements were in order some of them were not in order then the number of right shift operations for different managers could be different and they might take some intermediate values between zero and i. So in that case our overall running time will add up to some intermediate value. So what we are discovering here is something interesting. For the same input size n the running time can take a variety of values for insertion sort. So let me repeat this because this is important. For the same input size n t of n can take a variety of values which means it's not well defined. So if we were to plot the value of the running time as a function of the input the input size can be 1, 2, 3 and so on. These are discrete values because we are talking about array sizes here. So let's say we take n is equal to 10. For the same input size 10 if the array was sorted we might find that the running time is let's say this value whatever you know that particular value is given by the green point. But if the array was reverse sorted maybe the running time could be something like this. It's a higher value because if the array is reverse sorted all the managers are right shifting all the numbers on their left. So the overall running time is going to be higher. And in general the array will neither be fully sorted nor be exactly reverse sorted. It will be partially sorted. Some of the elements might be in order some of the elements might be out of order. So it depends on how much sorted the array already is. If it's already fully sorted then we need to do less work. In fact the least amount of work. If it's reverse sorted then insertion sort needs to do the most work. And to the extent that it is sorted we need to do a smaller amount of work because there would be a fewer number of right shift operations. And this is true of any input size even for an input size of say 4 we might have this being the running time if the array is already sorted. This being the running time if the array is reverse sorted. And there might be a bunch of other running times in the middle if the array is partially sorted. Now if we connect these different green dots together for different input sizes this line or curve in general is going to tell us what is the running time for each input size in the best case. That means in the scenario where the collective amount of work done on that input size is as small or as little as possible. So we're going to call this as the best case running time. Similarly if we were to connect all these red dots together we might find a different curve. And these red dots denote what is the maximum amount of work that our algorithm might have to do on an input of size n. And we're going to call this the worst case running time. And in general our running times are going to be between the worst case and the best case. So if we were to take the average of all of these white dots for any particular value of n. For example if I were to take n equal to 4 and if I were to just take the average of all possible running times for an input of size 4. I might get a value that is somewhere here shown by this yellow dot. Likewise for n equal to 10 I might get an average value that is let's say somewhere here. So I could take the average of the running times for each input size and then imagine connecting those together by means of a curve and that would give me the average case running time. So instead of talking about the running time in general for an input of size n. For algorithms like insertion sort we need to be more specific and say whether we are talking about the worst case running time, the average case running time or the best case running time. Because the running time in general is a highly variable quantity. So let's look at the average case. It's not just dependent on the size of the input or n. It is also dependent on the nature of the input. Whether the input in this case was already sorted, was reverse sorted or maybe it's partially sorted. Note that both the size of the input and the nature of the input are system independent factors. It doesn't matter what hardware we are running the algorithm on. If an array has a size of 100, it has a size of 100. If it's reverse sorted, it's reverse sorted. It doesn't matter what machine it's running on. So we still continue to ignore the system dependent factors. Even if we are talking about the worst case running time or the average case or the best case. So let's see what the free running times here are going to be for the insertion sort algorithm for a general input of size n. So going back to this expression for T , let's first calculate the best case running time for insertion sort. So in the best case, we know that the time spent by the ith manager is going to be 0 times C1. Because there are no right shift operations to be done. Plus some constant C2. And this is the amount of work involved in initializing that red index and copying out the value into the temporary variable only to copy it back. When you find that the red index is already pointing to a smaller number. So we are not going to scan the sub array on the left. We are just going to do a constant amount of work. And that completes the work at least for the ith manager. So we can write T in the best case as therefore the summation across all the managers of C2. A constant amount of work being done by each manager. So we can write C2. Not just by one manager. But every single manager doing the least amount of work. And so this is going to be C2 times n. Right. This is nothing but C2 added to itself n times. And so we can say the best case running time of insertion sort is theta of n. And this is when the array is already fully sorted. Now let's look at the worst case running time. In the worst case, the time spent by every single manager is going to be as high as possible. In other words, the number of ride shift operations done by each manager is going to be as high as the index that manager is responsible for. So the ith manager is going to do i number of ride shift operations. And so we are going to have T of n being equal to the summation of i varying from 0 to n minus 1 of i times C1 plus C2. So I can split this into two parts. I can say this is summation of i varying from 0 to n minus 1. We are going to add the work, the ride shift work done by each manager separately. And the constant initialization and you know copying the data into the temporary variable and back from the temporary variable as a separate term. This value we know is C2 times n. The first term can be written as C1 times the summation of i varying from 0 to n minus 1 of i. So in other words, what is the total number of ride shift operations that is done across the entire hierarchy that times the cost of doing a single ride shift. So the total number of ride shift operations that will be done across all the managers is going to be 0 for the leftmost manager, 1 for the next manager in sequence and so on all the way up to n minus 1. The last manager is going to right shift all n minus 1 of the numbers on the left side because he will find that the last number is smaller than all of them. And so the total number of ride shift operations in the worst case is going to be this and that times the cost of a single ride shift operation will be the amount of work being done across all the ride shifts and this plus C2 times n is going to be the total amount of work overall. Now the sum of the first n minus 1 numbers we keep seeing the same series reappearing over and over. So that we know is n times n minus 1 by 2 plus C2 times n. Now I do not want to spend too much time deriving an exact algebraic expression since we know that all that matters in asymptotic analysis is the dominant term. So what is the highest power of n that we are going to have in the result? So that means in the worst case my running time for insertion sort is theta of n square. Now let us look at the average case. What is the average case running time for insertion sort? So what will happen on average? So what happens on average really depends on what the average input is going to look like? What the average input of size n is going to look like? Is the typical input going to be sorted? Is it going to be reverse sorted? How much sorted is it already going to be on average? That's what matters for the average case. If we know in advance, for example, that the input is generally going to be almost sorted. Well, in that case, the average case running time is going to be pretty close to the best case running time. If we know that the typical input we are going to get is going to be almost reverse sorted, maybe not exactly, but close to being reverse sorted. In that case, the average case will be closer to the worst case running times. So we have to make some assumptions about what the typical input will be like in order to calculate the average case running time. So for our purposes here, we are just going to assume that on average, we are just going to have random numbers in the array with no particular ordering. In other words, if I were to pick any two numbers in the array, there is a 50-50 chance that the left number will be smaller than the right number or bigger than the right number. So in that case, let's look at what the expected number of right shift operations will be for an arbitrary manager. So an arbitrary manager who is focused on A is going to look on the left and is going to see I numbers on the left. So how many right shift operations will be expected for A ? Well, it depends on how many of those I numbers on the left are bigger than A . Now, if we are claiming that each number is as likely to be higher than A as to be lower than A , then roughly speaking, half of those numbers are going to be more than A and half of them will be smaller than A . So when I look on my left, on average, I am going to find half of the numbers being bigger than my number, which means on average, I am going to have to right shift half of the numbers on my left. So I can say that on average, the number of right shift operations that I will need to do are going to be I by 2 if I am looking at index I. So I can say that my average case running time in general is going to be the summation across all the managers of the amount of work done on average by an arbitrary manager, which is going to be I by 2 times C1 plus C2. So this is actually not that different from the summation we were doing for the worst case. The only change is I is replaced by I by 2. So if we were to go back to the calculations we did, if I want to find the average case running time, I would have an additional factor of 2 here. And that is going to end up with an additional factor of 2 in this series, which is going to end up with an additional 2 here. So that means that the average case running time of insertion sort is going to be the average of T of n. So that means that the average case running time of insertion sort is going to be the average of T of n. It is still going to be quadratic. So that means that the average case running time of insertion sort is going to be the average of the number of numbers that is more or less preferred than any other ordering. So that on average, I would expect if I am given any collection of say I numbers, I expect half of them to be smaller than a given number and half of them to be bigger than a given number. So to summarize, we wrote the code for insertion sort and when we analyzed it, we found that the running time was not purely a function of the input size. It was also a function of the nature of the input, how much ordered the input array already was. And so we had to split or be more specific about the running time by talking about the worst case running time, the average case running time and the best case running time, since just generically talking about the running time was not a meaningful thing to do. And this is markedly different from the much simpler analysis that we had for selection and bubble sort. Those two algorithms did not have a while loop, which introduces variability in how many times the loop is executed. If you remember there, we were actually looking on the right side instead of the left side. We were focused on the remaining elements beyond index i, trying to figure out which is the smallest among them and trying to get that smallest value into index i. So there was a predictable number of elements we were looking at on the right side. It didn't matter whether the original array was already sorted or reverse sorted or how much sorted it was. We were doing a fixed number of comparison operations in both selection and bubble sort. And so if we were to plot the graph of T of n versus n for selection and bubble sort, we would get no variability like this. There would be only a single value of T of n n for any given n. In other words, the best case, average case and worst case running times for selection and bubble sort, at least the way we wrote the code, all three of those values would be one and the same. But with insertion sort, we see that we have to split the running times into these three cases. The notes. the legs -->

Coding Insertion Sort:

- I am to write code to do insertion sort on an integer array `nums`:

```python
def insertion_sort(nums):
  pass
```

- I pretend to be an arbitrary lazy manager.
- I am given a prefix of the array (up to index `i`). I am to only sort this prefix/subarray (from index $0 \rightarrow i$). To do this, I depend on my subordinate.
  - I assume that my subordinate has already sorted a slightly smaller prefix of the original array (from index $0 \rightarrow i-1$)
  - I am going to try and extend that sorted region by one more element.
    - **How?** I will take the right most element of my subarray (`nums[i]`) and try to insert it into the appropriate position in my subordinate's subarray (from index $0 \rightarrow i-1$). And thereby, extend the sorted region by one more element.
      - **How?** I will copy this into a `temp` variable.

```python
def insertion_sort(nums):
  
    temp = nums[i]
```

- I will now start looking at all the elements in my subordinate's subarray (from index $0 \rightarrow i-1$) from right to left (using a variable called `j`). And as long as the value that `j` is pointing to is bigger than the value in the `temp` variable, I will keep right shifting those bigger values to the right. (to make space for the `temp` value to be **INSERTED** into its appropriate position.) Let's add the code for right to left scan:

```python
def insertion_sort(nums):

    temp = nums[i]
    j = i - 1             # j is initialized to the index of the last element in the subordinate's subarray
```

```python
def insertion_sort(nums):

    temp = nums[i]
    j = i - 1

    while __________ nums[j] > temp:    # As long as `nums[j]` is bigger than `temp`, 
      nums[j + 1] = nums[j]             # We keep right shifting `nums[j]` to the right.
      j -= 1                            # We also keep moving the `j` index one step to the left.
```

- There will come a point where `nums[j]` will be less than or equal to `temp`. And at this point, we must insert the `temp` value into the position to `j`'s right.

```python
def insertion_sort(nums):

    temp = nums[i]
    j = i - 1

    while __________ nums[j] > temp:
      nums[j + 1] = nums[j]
      j -= 1
    
    nums[j + 1] = temp                  # We insert the `temp` value into the position to `j`'s right.
```

- So now, I, the arbitrary manager, have successfully extended the sorted region by one more element.
- If each manager did this in a bottom up way, then the entire array should be sorted.

```python
def insertion_sort(nums):
  
  for i in range(1, len(nums)):          # Manager index `i` varies from 1 to n-1.
    
    # All the following arbitrary manager work is repeated by each manager in a bottom up order.
    temp = nums[i]
    j = i - 1
    
    while __________ nums[j] > temp:
      nums[j + 1] = nums[j]
      j -= 1
    
    nums[j + 1] = temp

  return nums                             # Return the sorted version of the original array.
```

- Subtle Scenario:
  - It is possible that as we are doing the right to left scan, every single number within the subordinate's subarray might be larger than the `temp` value. And in this case, the `j` index will jump off the left end of the array and its value will become `-1`. ILLEGAL INDEX!
  - So we must add an additional constraint that our `j` index needs to be greater than or equal to `0`.

```python
def insertion_sort(nums):
  
  for i in range(1, len(nums)):
    temp = nums[i]
    j = i - 1
    
    while j >= 0 and nums[j] > temp:      # j must always be greater than or equal to 0.
      nums[j + 1] = nums[j]
      j -= 1
    
    nums[j + 1] = temp

  return nums
```

- Note that this would work correctly even if the `temp` value that we are trying to insert is smaller than all the elements within the subordinate's subarray because let's say we have a subarray `[8, 7, 6, 5, 4, 3]` and we are trying to insert `2` into its appropriate position.
  - The `j` index will jump off the left end of the array and become `-1`.
  - And so we will end up copying the `temp` value into index `0`. (`nums[-1 + 1]` -> `nums[0]`)

#### DESIGN STRATEGY #1: Brute Force

- **Definition**: Directly based on the problem statement.
- **Pros/Cons**:
  - ✅: Straightforward 
  - ❌: May vary between individuals due to subjectivity.

#### DESIGN STRATEGY #2: Decrease and Conquer

- **Definition**: Reduce the problem size incrementally, typically by one, and solve the smaller problem.
- **Pros/Cons**:
  - ✅: Systematic approach to problem-solving.
  - ✅: Facilitates recursive thinking.
- **Types**:
  - **Top-Down Approach**: Do upfront work to reduce the problem size.
  - **Bottom-Up Approach**: Assume the smaller problem is solved and extend the solution.

### Decrease and Conquer Applied to Sorting

#### Top-Down Approach (Forward Decrease)

- **Method**: Find and place the smallest element at the beginning, reducing the problem size from $n$ to $n - 1$.
- **Process**:
  1. Find the minimum element in the array.
  2. Swap it with the element at index $0$.
  3. Recursively sort the subarray from index $1$ to $n - 1$.
- **Analogy**: "Lazy Manager" delegates the smaller subarray to a subordinate.
- **Resulting Algorithms**:
  - **Selection Sort**:
    - Repeatedly selects the minimum element and places it in its correct position.
  - **Bubble Sort**:
    - Bubbles up the smallest element through repeated swaps.

#### Bottom-Up Approach (Backward Decrease)

- **Method**: Assume the first $n - 1$ elements are sorted, then insert the $n$-th element into its correct position.
- **Process**:
  1. Treat the subarray from index $0$ to $n - 2$ as sorted.
  2. Take the element at index $n - 1$.
  3. Insert it into the sorted subarray to form a sorted array of size $n$.
- **Analogy**: "Lazy Manager" extends the solution provided by subordinates.
- **Resulting Algorithm**:
  - **Insertion Sort**:
    - Builds the final sorted array one item at a time by inserting elements into their correct positions.

### Insertion Sort Algorithm

#### Concept

- **Idea**: Insert each element into its appropriate position in the already sorted part of the array.
- **Process**:
  - Start from the second element.
  - For each element, compare it with elements on its left.
  - Shift larger elements to the right.
  - Insert the element at the correct position.

#### Efficient Implementation

- **Avoiding Excessive Swaps**:
  - Swapping elements is costly (three assignments per swap).
  - **Optimization**: Use shifting to reduce the number of assignments.
- **Procedure**:
  1. Store the element to be inserted in a temporary variable (`key`).
  2. Compare `key` with elements in the sorted subarray.
  3. Shift elements greater than `key` one position to the right.
  4. Insert `key` at the correct position.

#### Example

- **Initial Array**: `[2, 4, 5, 6, 7, 8, 3]`
- **Insertion of `3`**:
  - Compare `3` with `8`, `7`, `6`, `5`, `4`, shifting each to the right.
  - Stop when `2` is less than `3`.
  - Insert `3` after `2`.

### Time Complexity Analysis

- **Worst-Case Time Complexity**:
  - Occurs when the array is in reverse order.
  - **Operations**:
    - Comparisons and shifts for each element: up to $n - 1$.
    - Total operations: $O(n^2)$.
- **Best-Case Time Complexity**:
  - Occurs when the array is already sorted.
  - **Operations**:
    - One comparison per element.
    - Total operations: $O(n)$.
- **Average Time Complexity**:
  - **Operations**:
    - Depends on the initial order.
    - Total operations: $O(n^2)$.

### Key Points for Interviews

- **Decrease and Conquer Strategy**:
  - Understand both top-down and bottom-up approaches.
  - Recognize when to apply Decrease and Conquer in problem-solving.
- **Insertion Sort Characteristics**:
  - Simple to implement and understand.
  - Efficient for small or nearly sorted datasets.
- **Optimization Techniques**:
  - Reducing the number of operations by shifting instead of swapping.
  - Efficient memory usage by in-place sorting.
- **Algorithm Implementation Skills**:
  - Ability to write code for Insertion Sort.
  - Understanding the underlying mechanism and optimization.

### Summary

- **Decrease and Conquer** is a powerful strategy for algorithm design.
- **Insertion Sort** exemplifies the bottom-up approach of Decrease and Conquer.
- Efficient implementation reduces unnecessary operations.
- Proficiency in such algorithms is crucial for software engineering interviews.

---

