---
layout: post
title:  "Educational Codeforces Round 1"
date:   2018-02-20 08:00:00 +0000
permalink: 	/posts/:categories/:year/:month/:day/:title/
comments: true
categories: codeforces educational 
---

Welcome to the first post in a series of Educational Codeforces Round posts. I think these rounds are a great opportunity for people to get into competitive programming, but the lack of good editorials can make them a bit difficult to understand for complete beginners. Hopefully this series will fill some of the gaps the reader (and myself) might have. 

The goal is to do one of these once in a while and to include as much detail and explanation as possible. I intend to write solutions both in Python and C++. With the C++ solution I'll always aim for efficiency, whereas with the Python solution the goal is the write the most succinct solution. I might also write out a couple of solutions if I feel like it


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
# [Problem C - Nearest Vectors](http://codeforces.com/contest/598/problem/C)
# [Problem D - Igor in the Museum](http://codeforces.com/contest/598/problem/D)
# [Problem E - Chocolate Bar](http://codeforces.com/contest/598/problem/E)
# [Problem F - Cut Length](http://codeforces.com/contest/598/problem/F)