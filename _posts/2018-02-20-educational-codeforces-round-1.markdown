---
layout: post
title:  "Educational Codeforces Round 1"
date:   2018-02-20 08:00:00 +0000
permalink: 	/posts/:categories/:year/:month/:day/:title/
comments: true
categories: codeforces educational 
---

Welcome to the first post in a series of Educational Codeforces Round posts. I think these rounds are a great opportunity for people to get into competitive programming, but the lack of good editorials can make them a bit difficult to understand for complete beginners. Hopefully this series will fill some of the gaps the reader (and myself) might have. 

The goal is to do one of these once in a while and to include as much detail and explanation as possible. I intend to write solutions both in Python and C++. With the C++ solution I'll always aim for efficiency, whereas with the Python solution the goal is the write the most succinct solution. I might also write out a couple of solutions if I feel like it could contribute to understanding the problem.

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
The second solution has a similar complexity of \\( \mathcal{O}(m \times  \|s\|) \\) and doesn't use extra memory, but the inner loop that does the rotation is slightly more expensive, i.e. has a higher constant. This is however offset from not having to create a substring which is a very expensive operation.

The idea here is to to first find an index **middle** whose value is the position of the element that will be moved to the first position in the final string. We then keep 2 running indices - **start** going from **l** to **r**, which will define the portion of the string that's already in it's final place, and **next** starting from **middle**, which defines the position of the correct next element in the final string. We keep swapping the elements at **start** and **next** and increment both indices until **start** equals **middle**. At this point, we want to set **middle** again to be current position of the second index. Notice that this is the correct value of **middle** for the remainder of the string indexed by (**l + start**, **r**). We also want to make sure that if **next** reaches **r** we set it back to the current value of **middle** to make sure we don't go off the end of the string. We repeat this process until **start == next**. 

This probably sounds a lot more complicated than it is - I encourage you to have a look at the code and draw out the algorithm on a sheet of paper. 

Finally, we can actually abstract away the whole rotation algorithm by using the C++ STL **[rotate()](http://www.cplusplus.com/reference/algorithm/rotate/)** function. The above algorithm actually implements the same algorithm as **rotate()**. In the final code, I'm showing the version using the builtin function, but I've also left out the bits necessary to do our own rotation as comments.


{% highlight c++ %}
#include <iostream>
#include <string>
#include <algorithm>

using namespace std;

/* static void my_rotate(int start, int middle, int end, string &s) {
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

An obvious solution would be to brute force all combinations of vectors and find the angles between each combination in two nested loops, storing the smallest value. Note that this isn't going to work, because the complexity is \\( \mathcal{O}(n^2) \\), which will give us TLE on the given input size.

Insteaed, we can treat this as a sorting problem. We can find the angle each vector makes with the x axis and sort on that. Then, we can iterate through every consecutive pair of vectors and find the angles between them, storing the smallest result. Since the vectors are sorted, we're guaranteed that the angle each vector makes with it's two adjacent vectors is the smallest angle that vector makes with any other vector in the input data.

To find the angle between a vector and it's x axis, we can use the **atan2()** function.

### Solution (C++)
We're using a struct to store the angle the vector makes with the x axis, as well as the index of the vector for convenience and we use the builtin sort in C++. Everything else is as described above.

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
    int n, a, b, x, y;
    long double bestAns, currentAns;
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
    print(current, vectors[i][1], vectors[i+1][1])
    if (current < best and current != 0):
        best = current
        p1, p2 = vectors[i][1], vectors[i+1][1]

print(p1, p2)
{% endhighlight %}

# [Problem D - Igor in the Museum](http://codeforces.com/contest/598/problem/D)
# [Problem E - Chocolate Bar](http://codeforces.com/contest/598/problem/E)
# [Problem F - Cut Length](http://codeforces.com/contest/598/problem/F)