---
layout: post
title:  "Educational Codeforces Round 1"
date:   2018-02-20 08:00:00 +0000
permalink: 	/posts/:categories/:year/:month/:day/:title/
comments: true
categories: codeforces educational 
---

Welcome to the first post in a series of Educational Codeforces Round posts. I think these rounds are a great opportunity for people to get into competitive programming, but the lack of good editorials can make them a bit difficult towards understand for complete beginners. Hopefully this series will fill some of the gaps the reader (and myself) might have. 

I intend to write solutions both in Python and C++. With the C++ solution I'll always aim for efficiency, whereas with the Python solution the goal is the write the most succinct solution. I might also write out a couple of solutions if I feel like it could contribute to understanding the problem.

On that note, onto the contest! I encourage you to practice the round yourself [here](http://codeforces.com/contest/598/), before looking at the solutions. 

# [Problem A - Tricky Sum](http://codeforces.com/contest/598/problem/A)
This is a straightforward mathematical problem. We know that the sum of the first \\( n\\) natural numbers is: 

\\[1 + 2 + 3 + ... + n = {n (n + 1) \over 2} \\]

All we have to do is find that sum and then subtract from it all powers of 2 in that twice. We subtract once to nullify the existing terms in the sum, and subtract one more time to get the negative terms:

\\[-1 - 2 + 3 - 4 ... + n \\]

Let's try to evaluate the complexity of this solution. Given \\(t\\) numbers \\(n\\), we want to find the sum of the natural numbers up to \\(n\\), as well as and the sum of the powers of two that are smaller than or equal to \\(n\\). We can find the first sum in a constant time, using the formula above. For the second sum, we can find all terms in a loop which starts from 1 (\\(2^0)\\) and is multiplied by 2, giving us the next power \\(2^1, 2^2, ... ,2^{log_2(n)}\\). We can quickly see that the complexity of this loop would be \\( \mathcal{O}(\log_2(n)) \\). We have to run that \\(t\\) times, so we get overall complexity of \\( \mathcal{O}(t \log_2(n)) \\) As a quick reference, in the worst case, \\(t = 100\\) and \\(n = 10^9\\) and \\(log_2(10^9) \approx 33 \\). On average, we can assume that a judge can execute \\(10^8\\) operations per second. As you can see, our solution is more than quick enough for the given time limits and input sizes.

### Solution (C++)
One thing of note here is the input size for \\( n\\), which is \\( (1 \le n \le 10^9) \\) is too large for C++'s native **int** type. Given the formula above, the intermediate sum can be as large as \\( 10^{18} \\), which is too big even for the **long** type.  Luckily, the C++ **long long** type can be as big as \\(9 \times 10 ^ {18}\\), so it should be big enough to fit our calculations. 
{% highlight c++ %}
#include <iostream>

using namespace std;

int main() {
    int t;
    cin >> t;

    long long n;
    for (int i = 0; i < t; i++) {
        cin >> n;
        long long sum = n * (n + 1) / 2, power_sum = 0;
        for (long long j = 1; j <= n; j *= 2) {
            power_sum += j;
        }
        cout << sum  - power_sum * 2 << endl;
    }
    return 0;
}
{% endhighlight %}
### Solution (Python)
The Python solution is slightly less efficient as it calls the power operator \\(\log_2(n)\\) times and the complexity of the power operator is not constant. It depends on the size of the power and  \\( \mathcal{O}(\log_2(n)) \\), thus making the overall complexity \\( \mathcal{O}(\log_2^2(n)) \\). This is not going to be a problem here, because \\(\log_2(10^{9}) \approx 33 \\), but it is worth considering in the future. We could implement something similar to the above, but I prefer the shorter solution here. 

{% highlight python %}
import math

t = int(input())

for i in range(t):
    n = int(input())
    print(n*(n+1)//2 - 2*sum([2**x for x in range(int(math.log(n, 2))+1)]))
{% endhighlight %}

# [Problem B - Queries on a String](http://codeforces.com/contest/598/problem/B)
This problem requires us to find what an input string **s** would look like if we performed **m** partial right rotations on it, where each rotation is defined as a triplet (**l**, **r**, **k**) comprised of: the start index of the rotation - **l**, the end index of the rotation - **r** and the number of right shifts - **k**.

The first thing we notice is that if we perform **k** right shifts to a string of length **k**, we end up with the same string. Therefore, to make sure we don't do unnecessary shifts, we can set \\[k = k \bmod (r - l + 1) \\] After we have the correct **k**, it's easy to see that the first value of the new string(i.e. at index l would correspond to **r - k**. 



### Solution 1(C++)
Let's figure out how to use the above information to solve the problem. One way is to find a substring **s2** of **s** bound by **(l, r)** for the current triplet. We can then iterate through **s** (from **l** to **r**) and set the current character value to the **s2** value at the current index **j**. The initial value for **j** represents the position of the character in the original string but offset by **l** since the substring is indexed **(0, r - l)**, instead of **(l, r)**. We repeat that for every triplet. 


This first solution's complexity is \\( \mathcal{O}(m \times  \|s\|) \\), where \\(\|s\|\\) is the size of the input string **s**. This solution also requires  \\( \mathcal{O}(\|s\|)\\) extra memory for the substring.
{% highlight c++ %}
#include <iostream>
#include <string>

using namespace std;

int main() {
    string s;
    cin >> s;

    int m, l, r, k;
    cin >> m;
    for (int i = 0; i < m; i++) {
        cin >> l >> r >> k;
        int substring_size = r - l + 1;
        k = k % substring_size;
        string s2 = s.substr(l - 1, substring_size);
        for (int p = l - 1, j = substring_size - k; 
             p < r; 
             p++, j = (j + 1) % substring_size) {
            s[p] = s2[j];
        }
    }
    cout << s << endl;
    return 0;
}
{% endhighlight %}

### Solution 2(C++)
The second solution has a similar complexity of \\( \mathcal{O}(m \times  \|s\|) \\) and doesn't use extra memory, but the inner loop that does the rotation is slightly more expensive, i.e. has a higher constant. This is however offset by not having to create a substring which is a very expensive operation.

The idea here is to to first find an index **middle** whose value is the position of the element that will be moved to the first position in the final string. We then keep 2 running indices - **start** going from **l** to **r**, which will define the portion of the string that's already in it's final place, and **next** starting from **middle**, which defines the position of the correct next element in the final string. We keep swapping the elements at **start** and **next** and increment both indices until **start** equals **middle**. At this point, we want to set **middle** again to be current position of the second index. Notice that this is the correct value of **middle** for the remainder of the string indexed by (**l + start**, **r**). We also want to make sure that if **next** reaches **r** we set it back to the current value of **middle** to make sure we don't go off the end of the string. We repeat this process until **start == next**. 

This probably sounds a lot more complicated than it is. I encourage you to have a look at the code and draw out the algorithm on a sheet of paper. 

Finally, we can actually abstract away the whole rotation algorithm by using the C++ STL **[rotate()](http://www.cplusplus.com/reference/algorithm/rotate/)** function. The above algorithm actually implements the same algorithm as **rotate()**. In the final code, I'm showing the version using the builtin function, but I've also left out the bits necessary to do our own rotation as comments.


{% highlight c++ %}
#include <iostream>
#include <string>
#include <algorithm>

using namespace std;

/* inline void my_rotate(int start, int middle, int end, string &s) {
    if (middle == end) return;
    int next = middle;
    while (start < next) {
        swap(s[start++],  s[next++]);
        if (next == end) next = middle;
        else if (start == middle) middle = next;
    }
} */

int main() {
    string s;
    cin >> s;

    int m, l, r, k;
    cin >> m;
    for (int i = 0; i < m; i++) {
        cin >> l >> r >> k;
        k = k % (r - l + 1);
        rotate(s.begin() + l - 1, s.begin() + r - k, s.begin() + r);
        // my_rotate(l - 1, r - k, r, s);
    }
    cout << s << endl;
    return 0;
}
{% endhighlight %}
### Solution (Python)
To rotate a string **s** in Python, we can use slicing operations to construct a new string **s = s[-k:] + s[:-k]**, where **k** is the position of the element that ends up first in the new string. Adjusting for (**l**, **r**, **k**), we have the following solution:

{% highlight python %}
s = input()
m = int(input())

for i in range(m):
    (l, r, k) = [int(x) for x in input().split()]
    k = k % (r - l + 1)
    s = s[:l-1] + s[r-k:r] + s[l-1:r-k] + s[r:]

print(s)
{% endhighlight %}
# [Problem C - Nearest Vectors](http://codeforces.com/contest/598/problem/C)
We're given a set of vectors in a 2D plane, all starting at the origin and we're asked to find a pair of vectors with smallest non-oriented(positive) between them. 

An obvious solution would be to brute force all possible pairs of vectors and find the angles between each combination in two nested loops, storing the smallest value. Note that this isn't going to work, because the complexity is \\( \mathcal{O}(n^2) \\), which will give us TLE on the given input size.

Instead, we can treat this as a sorting problem. We can find the angle each vector makes with the x axis and sort on that. Then, we can iterate through every consecutive pair of vectors and find the angles between them, storing the smallest result. Since the vectors are sorted, we're guaranteed that the angle each vector makes with it's two adjacent vectors is the smallest angle that vector makes with any other vector in the input data.

The overall complexity here is dominated by the sorting term, and C++ **sort()** is \\(\mathcal{O}(n \log(n)) \\).

To find the angle between a vector and it's x axis, we can use the **atan2()** function.

### Solution (C++)
We're using a struct to store the angle the vector makes with the x axis, as well as the index of the vector for convenience and we use the builtin sort in C++. Another thing to keep in mind is that we need to lose **long double** type, otherwise we lose precision.

{% highlight c++ %}
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

const long double PI = atan(1) * 4;

struct PointVector {
    long double angle;
    int i;
};

bool cmp(PointVector a, PointVector b) {
    return a.angle < b.angle;
}

int main()
{
    int n, a, b;
    long double bestAns, currentAns, x, y;
    cin >> n;
    vector<PointVector> vectors(n);
    for (int i = 0; i < n; i++) {
        vectors[i].i = i + 1;
        cin >> x >> y;
        vectors[i].angle = atan2(y, x);
    }

    sort(vectors.begin(), vectors.end(), cmp);
    
    bestAns = (2 * PI - abs(vectors[0].angle) - abs(vectors[n - 1].angle));
    a = vectors[0].i; b = vectors[n-1].i;
    for (int i = 0; i < n - 1; i++) {
        currentAns = vectors[i + 1].angle - vectors[i].angle;
        if (currentAns < bestAns) {
            bestAns = currentAns;
            a = vectors[i].i;
            b = vectors[i + 1].i;
        }
    }
    cout << a << " " << b << endl;

    return 0;
}
{% endhighlight %}
### Solution (Python)
Here is an equivalent Python implementation. Note, that calculating the angle might cause issues with the judge system. We might need a more numerically stable method for finding the angles (ideally one such that doesn't use division). 
{% highlight python %}
import math

n = int(input())

vectors = []
for i in range(n):
    x, y = map(int, input().split())
    vectors.append((math.atan2(y, x), i + 1))
vectors.sort()

best = 2 * math.pi - abs(vectors[0][0]) - abs(vectors[n-1][0])
p1, p2 = vectors[0][1], vectors[n-1][1]
for i in range(n - 1):
    current = vectors[i+1][0] - vectors[i][0]
    if (current < best and current != 0):
        best = current
        p1, p2 = vectors[i][1], vectors[i+1][1]

print(p1, p2)
{% endhighlight %}

# [Problem D - Igor in the Museum](http://codeforces.com/contest/598/problem/D)
We're given a rectangular **n x m** grid with passable **'.'** and impassable **'*'** cells. Given **k** starting positions where we're guaranteed to start at a passable cell, we want to find the maximum number of impassable cells when can access for each starting position.

This is a classic Depth-first search problem. We'll explore all possible cells from a starting position and record the final count back to stdout. Since we use our **visited** array to store which nodes of the graph have been visited, we are guaranteed that every node will be visited at most once. Thus the overall complexity is \\( \mathcal{O}(m \times n) \\).
### Solution (C++)
{% highlight c++ %}
#include <cstdio>

using namespace std;

char grid[1001][1001];
int visited[1001][1001];
int ans[1000001];
int n, m, k, component = 1;

int count(int x, int y) {
    if (visited[x][y]) return 0;
    if (grid[x][y] == '*') return 1;
    if (grid[x][y] == '.') visited[x][y] = component;
    return count(x-1, y) + count(x, y-1) + count(x+1, y) + count(x, y+1);
}

int main()
{
    scanf("%d %d %d\n", &n, &m, &k);
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            scanf("%c", &grid[i][j]);
        }
        scanf("\n");
    }
    int x, y;
    while (k--) {
        scanf("%d %d\n", &x, &y);
        if (!visited[x][y]) {
            ans[component] = count(x, y);
            component++;
        }
        printf("%d\n", ans[visited[x][y]]);
    }
    return 0;
}
{% endhighlight %}
### Solution (Python)
I actually struggled to find a Python solution here that is able to run within the given time limit. I think the function calls in the recursion are very expensive. It looks like the problem was designed to be solved by faster languages, given the judge time limit, but I could be wrong. Please comment below if you have an answer!
{% highlight python %}
ddef count(x, y):
    if visited[x][y] != 0: return 0
    if grid[x][y] == '*': return 1
    visited[x][y] = component
    return count(x-1, y) + count(x, y-1) + count(x+1, y) + count(x, y+1)

n, m, k = map(int, input().split())

grid = []
for _ in range(n):
    grid.append(input())

ans, visited, component = {0: 0}, [[0] * (m) for _ in range(n)], 1
while(k):
    x, y = map(int, input().split())
    x, y = x-1, y-1
    if visited[x][y] == 0:
        ans[component] = count(x, y)
        component = component + 1
    k = k - 1
    print(ans[visited[x][y]])
{% endhighlight %}
# [Problem E - Chocolate Bar](http://codeforces.com/contest/598/problem/E)
Let's try to define a recurrence relation which solves this problem. Given a whole **n x m** piece, we can only split in one direction at a time, i.e. either vertically or horizontally. If we split along the **n** dimension, the cost of splitting is **m x m**. Similarly, if we split along the **m** dimension, the cost of splitting is **n x n**. Finally, when we perform a split, we can take **i** squares from one of the splits and **k - i** squares from the other split.

Now, let's define a function \\(f(n, m, k)\\), where **f** is the minimum cost required to get **k** squares from a **n x m** piece. We can find the minimum cost by exploring all possible vertical(**n**) and horizontal(**m**) splits and for each split we want to explore every possible configuration **(i, k - i)** of taking **k** squares. If we define \\(f_{h}(n, m, k)\\)  as minimum cost required to get **k** squares from **n x m** piece if we split horizontally and \\(f_v(n, m, k)\\) if we split vertically, we end up with the following recurrence relation: 

$$f(n, m, k) = 
\begin{cases}
min(f_{h}(n, m, k), f_{v}(n, m, k)) & \text{if $n \times m \neq k$} \\
0 & \text{if $n \times m = k$} \\
0 & \text{if $k = 0$} 
\end{cases}$$
$$f_{h}(n, m, k) = \forall_{i \in [0,n]} \forall_{j \in [1,k]} min(f(n, i, j), f(n, m - i, k - j ))$$
$$f_{v}(n, m, k) = \forall_{i \in [0,m]} \forall_{j \in [1,k]} min(f(i, m, j), f(n - i, m, k - j ))$$

We can see that we can exhaust all split combinations using two nested loops. Another realization is that we can leverage symmetry to only check half of all possible splits - the splits **(i, n-i)** and **(n-i, i)** give us the same pieces.

Finally, the problem has an optimal substructure and overlapping subproblems. This means we can use dynamic programming to improve its currently exponential complexity. Instead of repeating computations for a given combination of \\(f(n, m, k)\\), we can use memoization store each possible value of the function, giving us an overall complexity of \\( \mathcal{O}(t \times n \times m \times k) \\).

After all the work we've done, the solutions below are trivial:

### Solution (C++)
{% highlight c++ %}
#include <iostream>

using namespace std;

int dp[31][31][51];

int cost(int n, int m, int k) {

    if (dp[n][m][k] || k == 0 || n * m == k) return dp[n][m][k];

    int bestMin = 1e9;
    for (int i = 1; i <= n - i; i++) {
        for (int j = 0; j <= k; j++) {
            bestMin = min(bestMin, cost(i, m, j) + cost(n - i, m, k - j) + m * m);
        }
    }

    for (int i = 1; i <= m - i; i++) {
        for (int j = 0; j <= k; j++) {
            bestMin = min(bestMin, cost(n, i, j) + cost(n, m - i, k - j) + n * n);
        }
    }

    return dp[n][m][k] = bestMin; 
}

int main() {
    int t;
    cin >> t;
    int n, m, k;
    while(t--) {
        cin >> n >> m >> k;
        cout << cost(n, m, k) << endl;
    } 
}
{% endhighlight %}

### Solution (Python)
The equivalent Python solution is, as expected, much slower. We can make an interesting observation on performance however - if we define **c** as **1e9** like we did in the C++ solution, we won't pass the judge time limit. This is because in Python **1e9** is a **float,** whereas in C++ it gets truncated to an an **int**. Because **c** ends up being a **float**, all **int** values that get compared to it get coerced to float and operations on floating point numbers are much slower. 
{% highlight python %}
t = int(input())

dp = [[[0 for i in range(51)] for j in range(31)] for k in range(31)]

def cost(n, m, k):
    if (dp[n][m][k] or k == 0 or n * m == k): return dp[n][m][k]
    c = 10**9
    for i in range(1, n // 2 + 1):
        for j in range(k + 1):
            c = min(c, cost(i, m, j) + cost(n - i, m, k - j) + m * m)
    for i in range(1, m // 2 + 1):
        for j in range(k + 1):
            c = min(c, cost(n, i, j) + cost(n, m - i, k - j) + n * n)
    dp[n][m][k] = c
    return c

for _ in range(t):
    n, m, k = map(int, input().split())
    print(cost(n, m, k))
{% endhighlight %}

# [Problem F - Cut Length](http://codeforces.com/contest/598/problem/F)
This is one of the more geometrical problems in Codeforces. Given a simple (without self-intersections) n-gon, and m lines, we need to find the length of the common part of each line with the n-gon. 

To begin with, we want to make sure we have a consistent ordering of the polygon vertices. We're going to treat each vertex as a position vector(i.e. starting from the origin O).
We know that the cross product between two vectors gives us the oriented area of the paralellogram defined by the two vectors.  Similarly, we can use the shoelace formula to find the oriented area of the polygon. We can use the sum of the cross products of all position vectors, which will give us the doubled oriented area. We don't really care about the actual area, but it's sign, which will tell us whether or not our vertices are given clockwise or counter-clockwise. We make sure that if the doubled oriented area is negative, then we reverse the order of the vertices so that they are always ordered clockwise.

Let's think about the problem itself. We have a line and we want to find all of its intersections with the polygon. The parametric equation of a line tells us that PQ = OP + t * (OQ - OP). If we can somehow find the value of t that represents the magnitute of all of the intersections of the line with the polygon, we would have our solution.

Lets iterate through every pair of adjacent vertices. We have a line, defined by the points P and Q, and two adjacent vertices, A and B. The vectors PA and PB are given by OA - OP and OB - OP respectively, and the vector PQ is given by OQ - OP. We know that the cross product sign tells us which side the first vector lies on with respect to the second vector. Therefore, we can use that information to determine if AB intersects PQ by checking the sign of the cross products of both PA and PB with PQ. If the signs are different(or 0, when the vectors are collinear), it means that AB intersects PQ and not otherwise. 

We can find an equation for t at the intersection, using some vector algebra:

Using parametric equation of a line, we have:

OP + t * (OQ - OP) for the line defined by the points P and Q, defining the given line.
OA + r * (OB - OA) for the line defined by the points A and B, defining a polygon edge.

The lines intersect only when:

OP + t * (OQ - OP) = OA + r * (OB - OA)

Lets cross both sides by (OB - OA):

(OP + t * (OQ - OP)) X (OB - OA) = (OA + r * (OB - OA)) X (OB - OA)

Since v X v = 0, we can simplify and find this equation for t at the intersection:

t = (OA - OP) X (OQ - OP) / (OQ - OP) X (OB - OA)

Great! We're almost there. Finally, we need to consider the order of the intersections and how to combine them together. For each intersection we store the t value and the type. There are 6 types of intersections, defined by the cross product signs we calculated earlier. We define an intersection type between AB and PQ as the difference between the sign of the cross product of BP with PQ and AP with PQ. The net sum of all intersection types should always be 0. This is because for all intersections where AB and PQ aren't collinear, there is an even number of intersections, such that for each "opening" intersection there is a "closing" one. Therefore if, we sort the intersections by t and intersection type, we can make sure that we know when two intersection points define an edge inside or outside of the polygon. 

This probably sounds very obfuscated. I encourage you to look at the code below and draw out a couple of examples - hopefully it will make all of this clearer.
 
### Solution (C++)
{% highlight c++ %}
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

const long double eps = 1e-9;

inline int sign(long double num) {
    if (num > eps) return 1;
    if (num < -eps) return -1;
    return 0;
}

struct Point {
    long double x, y;
    Point() {}
    Point(long double x, long double y): x(x), y(y) {}
    Point operator + (const Point &other) const {
        return Point(x + other.x, y + other.y);
    }

    Point operator - (const Point &other) const {
        return Point(x - other.x, y - other.y);
    }

    long double operator ^ (const Point &other) const {
        return x * other.y - y * other.x;
    }

    long double length() const {
        return hypotl(x, y);
    }
};

void solve(vector<Point> polygon, Point p, Point q) {
    int n = polygon.size();

    vector<pair<long double, int>> intersectionPairs;

    for (int i = 0, j = i + 1; i < n; i++, j = (j + 1) % n) {
        int startSign = sign((polygon[i] - p) ^ (q - p));
        int endSign = sign((polygon[j] - p) ^ (q - p));
        if (startSign == endSign) continue;
        long double t = ((polygon[i] - p) ^ (polygon[i] - polygon[j])) / ((q - p) ^ (polygon[i] - polygon[j]));
        intersectionPairs.push_back(make_pair(t, endSign - startSign));
    }

    sort(intersectionPairs.begin(), intersectionPairs.end());

    long double totalT = 0, previousT = 0;
    int count = 0;
    for (auto intersectionPair : intersectionPairs) {
        if (count > 0) totalT += intersectionPair.first - previousT;
        previousT = intersectionPair.first;
        count += intersectionPair.second;
    }

    cout << totalT * (q - p).length() << endl;
}

int main() {
    int n, m;
    cout << fixed << setprecision(12);
    cin >> n >> m;

    vector<Point> polygon(n);
    for (int i = 0; i < n; i++) {
        cin >> polygon[i].x >> polygon[i].y;
    }

    long double area = 0;
    for (int i = 0, j = i + 1; i < n; i++, j = (j + 1) % n) {
        area += polygon[i] ^ polygon[j];
    }

    if (area < 0) reverse(polygon.begin(), polygon.end());

    for (int i = 0; i < m; i++) {
        Point p, q;
        cin >> p.x >> p.y >> q.x >> q.y;
        solve(polygon, p, q);
    }
}
{% endhighlight %}
### Solution (Python)
{% highlight python %}
import math

eps = 1e-9

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, v):
        return Vector(self.x + v.x, self.y + v.y)
    
    def __sub__(self, v):
        return Vector(self.x - v.x, self.y - v.y)
    
    def length(self):
        return math.hypot(self.x, self.y)

def sign(n):
    if n > eps: return 1
    if n < -eps: return -1
    return 0

def cross(a, b):
    return a.x * b.y - a.y * b.x

def solve(polygon, p, q):
    intersections = []
    for (a, b) in zip(polygon, polygon[1:] + polygon[:1]):
        ss = sign(cross(a - p, q - p))
        es = sign(cross(b - p, q - p))

        if ss == es: continue

        t = cross(a - p, a - b) / cross(q - p, a - b)
        intersections.append((t, es - ss))
    intersections = sorted(intersections)

    total_t, previous_t, count = [0] * 3
    for t, order in intersections:
        if (count > 0): total_t += t - previous_t
        previous_t = t
        count += order

    print(total_t * (q - p).length())
    
n, m = map(int, input().split())

polygon = []
for i in range(n):
    x, y = map(float, input().split())
    polygon.append(Vector(x, y))
area = sum(map(lambda x: cross(x[0], x[1]), zip(polygon, polygon[1:] + polygon[:1])))
if (area < 0): polygon.reverse()

for i in range(m):
    x1, y1, x2, y2 = map(float, input().split())
    solve(polygon, Vector(x1, y1), Vector(x2, y2))

{% endhighlight %}