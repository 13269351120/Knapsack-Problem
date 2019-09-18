# Knapsack-Problem
Knapsack Problem Conclusion

### 背包问题
- 01背包
- 完全背包
- 混合背包
- 二维费用
- 分组背包
- 背包问题求方案数
- 背包问题的方案
- 有依赖的背包

> 常用的STL函数：  
求和：accumulate(vec.begin(), vec.end(), START_VALUE);  
求最大值：auto it = max_element(vec.begin(), vec.end()); 最大值为：*it  


#### 01背包   
01背包是一个复杂的NP问题，当数据规模比较小的时候，可以考虑用dfs来做。从dfs的角度，也可以让我们对整个过程更加的熟悉。  
下面给出dfs的伪代码：  
```python
def dfs(s, cur_w, cur_v):
    ans = max(ans, cur_v)
    if s > N: return 
    for i in range (s, N+1):
        if (cur_w + w[i] <= W):
            dfs(i+1, cur_w+w[i], cur_v+v[i])
            
def Knapsack01(w, v):
    ans = 0
    dfs(1, 0, 0)
    return ans
```  
在进行dfs的时候，我们清楚的看到，每一件物品有取和不取两种情况。  

最经典的背包问题，核心在于牢记f(i,w)的语义  
> 情况1: f(i,w)表示使用前i件物品，且空间为w时，最多可以产生的价值V，这里不要求空间全部用完。这种语义的好处是在最后返回的时候，只需要返回f(n,W)就可以了。前提是`f(0,0~W)=0`初始化。这种语义在求解具体方案的个数的时候可能会有有些歧义。

> 情况2: f(i,w)表示使用前i件物品，且恰好使用了w的空间，最多可以产生的价值V，这种方法的好处是语义比较清晰，前提是`f(0,0)=0` 且 `f(0,1~W)=-INF`。  

不过殊途同归，弄懂了原理都是一样的。  
状态转移方程：  
`f(i, w) = max(f(i-1, w), f(i-1, w-w[i]) + v[i])`

具体的代码：
```cpp 
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;
int dp[N][N];

int main() {
    int n, V;
    cin >> n >> V;
    for (int i = 1; i <= n; i++) {
        int v, w;
        cin >> v >> w;
        for (int j = 0; j <= V; j++) {
            dp[i][j] = dp[i-1][j]; //这句话很重要
            if (j >= v) dp[i][j] = max(dp[i][j], dp[i-1][j-v] + w);
        }
    }
    cout << dp[n][V] << endl;
    return 0;
}
```
**注意**：我在写惯了优化后的滚动数组的方案后，突然要写一个二维数组的形式吗，我很自然的忽略了 `dp[i][j] = dp[i-1][j];` 因为在滚动数组的时候，这句话是默认的。然而在二维数组中，需要自己赋值来保证语义。如下代码是错误的实例： 
```cpp
        for (int j = 0; j <= V; j++) {
            // dp[i][j] = dp[i-1][j]; //这句话很重要
            if (j >= v) dp[i][j] = max(dp[i][j], dp[i-1][j-v] + w);
        }
```


优化方案：  
在使用二维数组的时候我们发现了f(i,w)的状态仅仅与上一行的状态发生关联，而与前面若干行的状态无关，所以可以降纬，但是需要注意的是：如果还是按照顺序进行求解，f(i,w) = f(i-1, w-w[i])会导致覆盖了之前的值，所以需要从后往前进行计算。这也是滚动数组的一个最最基本的优化。  
下面是实现的代码：
```cpp
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;
int dp[N];

int main() {
    int n, V;
    cin >> n >> V;
    for (int i = 1; i <= n; i++) {
        int v, w;
        cin >> v >> w;
        for (int j = V; j >= 1; j--) {
            if (j >= v) dp[j] = max(dp[j], dp[j-v] + w);
        }
    }
    cout << dp[V] << endl;
    return 0;
}
```
**总结**：理解01背包问题及其滚动数组优化方案是解决整个背包问题体系的重要一步，01背包要注意的有：分清楚dp`[N][N]`的语义，是物体体积恰好为N的，还是背包体积为N，允许物体总体积不为N。不同的语义造就不同的最终结果。尤其要注意的是，在考虑物体体积恰好为N的这种语义的情况，最后要遍历求出最大值，如果还需要求方案的个数，则不能仅仅返回某个背包体积下的方案数，而是要把所有等于最大值的方案数累加。具体例题在`背包问题求方案数`详细展开。
#### 完全背包  
完全背包问题是01背包问题的拓展，所有物体可以取无数次。在01背包的优化的时候，我们依稀的记得，为什么滚动数组体积要从后往前遍历。那是因为要避免在算后面的值得时候，前面的值已经被修改过了，即`dp[j] = max(dp[j], dp[j-v] + w)`，而在01背包的语义下，一旦前面的值被修改，意味着这件物品已经被选择过了，所以需要从后往前遍历保证前面的还没有被修改过。  
但是考虑完全背包的背景，我们恰恰希望可以在该件物品选择后，依然可以选择该件物品，所以多重背包就是01背包优化方案从前往后的版本。 
具体代码如下：  
```cpp
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

int main() {
    int n, V;
    cin >> n >> V;
    vector<int> dp(V+1, 0);
    for (int i = 0; i < n; i++) {
        int v, w;
        cin >> v >> w;
        for (int j = v; j <= V; j++) {
            dp[j] = max(dp[j], dp[j-v]+w);
        }
    }
    cout << dp[V] << endl;
}
```

#### 多重背包
多重背包看上去是 介于 01背包和 完全背包的问题的过渡环节。  
> 其中一种思路是将物品展开，转化成01背包问题，比如说体积为v，价值为w的物品有5件，那么展开来就是`(v,w),(v,w)...(v,w)`每件物品都只能有取和不取两种情况。  

> 另一种思路是在取得策略上遍历，还是上述例子，我们可以遍历取0件，1件，2件...5件该物品。  
这个思路反应出背包问题一个重要的思想：  
第一重循环： 遍历物品  
第二重循环： 遍历限制（如果限制不止一个，比如包的大小有限制，包的重量也有限制，这样就会导致多重循环    
第三重循环： 遍历策略   

有了这样的思路我们就可以写出代码：
- 思路一：
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
struct Good {
    int v, w;  
};
int main() {
    int n, V;
    cin >> n >> V;
    vector<int> dp(V+1, 0);
    vector<Good> goods;
    for (int i = 1; i <= n; i++) {
        int v,w,s;
        cin >> v >> w >> s;
        for (int j = 0; j < s; j++) {
            goods.push_back({v,w});
        }
    }
    for (int i = 0; i < goods.size(); i++) {
        for (int j = V; j >= goods[i].v; j--) {
            dp[j] = max(dp[j], dp[j-goods[i].v] + goods[i].w);
        }
    }
    cout << dp[V] << endl;
}
```
- 思路二：
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
    int n, V;
    cin >> n >> V;
    vector<int> dp(V+1, 0);
    for (int i = 1; i <= n; i++) {  //第一重循环： 遍历物品  
        int v,w,s;
        cin >> v >> w >> s;
        for (int j = V; j >= v; j--) { //第二重循环： 遍历限制
            for (int k = 1; k <= s && j - k * v >= 0; k++) { //第三重循环： 遍历策略   
                dp[j] = max(dp[j], dp[j-k*v] + k * w);
            }
        }
    }
    cout << dp[V] << endl;
}
```

**优化的方案**  
优化的思路：我们看到第一种方法里面，我们将一件物品可以出现多少次进行展开，放入一个数组里去，从而转化为了01背包问题，从这个思路出发，如果我们可以通过不那么简单的拆分来表示一个数。  
具体例子：  
> 13 = 1 + 1 + ... + 1 （一共13个数）  
> 13 = 1 + 2 + 4 + 6   （一共4个数）  
然而，4个数的组合就可以完美的组成1~13种的任意数字，这是2的n次方一个很重要的性质。通过这一想法，可以将粗浅思路的01背包，优化组成的物品个数。  
下面是实现代码：
```cpp
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

struct Good {
    int v,w;  
};

int main() {
    int N, V;
    cin >> N >> V;
    vector<int> dp(V+1, 0);
    vector<Good> goods;
    for (int i = 0; i < N; i++) {
        int v, w, s;
        cin >> v >> w >> s;
        for (int j = 1; j <= s; j *= 2) {
            s -= j;
            goods.push_back({j * v, j * w});
        }
        if (s > 0) {
            goods.push_back({s * v, s * w});
        }
    }
    for (int i = 0; i < goods.size(); i++) {
        for (int j = V; j >= goods[i].v; j--) {
            dp[j] = max(dp[j], dp[j-goods[i].v] + goods[i].w);
        }
    }
    cout << dp[V] << endl;
    return 0;
}
```

值得注意的是：在二进制拆分后，可能会剩下一个余数，比如说刚刚` 13 = 1 + 2 + 4 + 6` 中的这个6，这也要考虑进去。


**待做**：
> 使用单调队列的优化方式


#### 混合背包 
混合背包在掌握了以上 01背包，完全背包， 多重背包解决方案后，显得轻而易举。  
#### 二维费用的背包
二维费用的背包问题，就是根据：
> 第一重循环： 遍历物品  
第二重循环： 遍历限制（如果限制不止一个，比如包的大小有限制，包的重量也有限制，这样就会导致多重循环    
第三重循环： 遍历策略   

在限制的时候，升维数，多重遍历即可。如下代码：
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
const int S = 1010;
int dp[S][S];
int main() {
    int N, V, M;
    cin >> N >> V >> M;
    for (int i = 0; i < N; i++) {
        int v, m, w;
        cin >> v >> m >> w;
        for (int j = V; j >= v; j--) {
            for (int k = M; k >= m; k--) {
                dp[j][k] = max(dp[j][k], dp[j-v][k-m] + w);
            }
        }
    }
    cout << dp[V][M] << endl;
    return 0;
}
```

#### 分组背包
分组背包问题是一个 以组为单位的 01背包问题，与01背包不同的是，01背包已经确定了每一个物品的信息，而分组背包需要在一个组里面去挑选，用一个循环遍历就可以解决。  
下面是代码：
```cpp
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

const int N = 1010;
int dp[N];

int main() {
    int n,V;
    cin >> n >> V;
    for (int i = 0; i < n; i++) {
        int s;
        cin >> s;
        vector<pair<int,int>> goods(s, {0,0});
        for (int j = 0; j < s; j++) {
            cin >> goods[j].first >> goods[j].second;  //first: v second:w
        }
        for (int j = V; j >= 0; j--) {
            for (auto good : goods) {
                if (j >= good.first) dp[j] = max(dp[j], dp[j-good.first] + good.second);
            }
        }
    }
    cout << dp[V] << endl;
    return 0;
}
```
这里有一个小小的陷阱，刚开始做的时候，我不假思索的把循环写成这样：
```cpp
for (int i = 0; i < n; i++) { //考虑第i小组
    for (int j = 0; j < s; j++) { //第i小组内，分别考虑s件物品
        for (int k = V; k >= 0; k--) { //背包容量遍历
                
            }
        }
    }
}
```
这样的话就会导致错误，因为我们要遵循的是：  
> 第一重循环： 遍历物品  
第二重循环： 遍历限制（如果限制不止一个，比如包的大小有限制，包的重量也有限制，这样就会导致多重循环    
第三重循环： 遍历策略  

这里将策略放在了中间，应该是将条件全部限制好：  
**1.考虑第i件物品，在体积为j的情况，再分别比较放几件**  
**2.考虑第i组，在体积为j的情况，再分别比较组内哪件**   

##### 那么为什么要以这样的循环顺序呢？如果将`遍历策略`和`遍历限制`的顺序换一下可以吗？  
> 仔细一想，假设第i小组中考虑第1件物品，然后进行背包容量的遍历，这样求出来的是，各个背包容量在考虑装不装第1件物品时候的情况；紧接着我们考虑同一个小组里的第2件物品，此时我们不能保证第2件和第1件物品互斥。  

> 如果是按照先`遍历限制`后`遍历策略`，这样就变成在同个外界条件下，不同策略之间的PK，这样可以导致不同策略之间是互斥的。所以记住一句话，不同策略之间的PK的前提是要在外界条件完全一致的情况下才有意义。  



#### 背包问题求方案数
背包问题求方案无非就是把状态记录下来，从语义看`dp[i][j]`有两种：  
- 表示考虑到第i件物品，体积恰好为j的时候的最大价值，其初始值`dp[0][0] = 0`,`dp[0][1~V]=-INF`,这就保证了所有的状态都是从`dp[0][0]`转移过去的。
- 表示考虑到第i件物品，体积为j，但不要求盛满最大价值,其初始值`dp[0][0~V] = 0`，如果一个状态可以从`dp[0][0]`转移过去，那么一定可以经过一定的偏移`dp[dx][0]`转移。    
显然第二种语义更加的广泛，也更加的常用，一般我们直接返回dp[V]就完事了。在求解背包问题的方案数的情况时，我们先从最清晰的语义，第一种语义写代码。  

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits.h>

using namespace std;

const int N = 1010;

const long long mod = 1e9+7;

struct state {
    int value, count;   
    state() {value = INT_MIN; count = 0;} //初始化
    state(int v, int c) : value(v), count(c) {}
};

state dp[N];

int main() {
    int n, V;
    cin >> n >> V;
    dp[0] = {0, 1};   //初始化
    for (int i = 0; i < n; i++) {
        int v, w;
        cin >> v >> w;
        for (int j = V; j >= v; j--) {
            int count = 0;
            int value = max(dp[j].value, dp[j-v].value+w);
            if (value == dp[j].value) count += dp[j].count;
            if (value == dp[j-v].value + w) count += dp[j-v].count;
            count %= mod;
            dp[j] = {value, count};
        }
    }
    //求解出所有状态后需要先求出最大值，然后把所有最大值的情况叠加起来
    int max_value = INT_MIN;
    for (int i = 1; i <= V; i++) {
        max_value = max(dp[i].value, max_value);
    }
    int res = 0;
    for (int i = 1; i <= V; i++) {
        if (max_value == dp[i].value) {
            res += dp[i].count;
            res %= mod;
        }
    }
    cout << res << endl;
    return 0;
}
```

上面的代码看上去要比第二种复杂：  
> 1）初始化的时候需要注意  
> 2）求解完后有一个遍历求和的过程  

下面是第二种语义：  
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits.h>

using namespace std;

const int N = 1010;

const long long mod = 1e9+7;
struct state {
    int value, count;   
    state() {value = 0; count = 1;} //全部统一初始化，从任意状态都可以转化，不一定非从dp[0]转化
    state(int v, int c) : value(v), count(c) {}
};

state dp[N];

int main() {
    int n, V;
    cin >> n >> V;
    for (int i = 0; i < n; i++) {
        int v, w;
        cin >> v >> w;
        for (int j = V; j >= v; j--) {
            int count = 0;
            int value = max(dp[j].value, dp[j-v].value+w);
            if (value == dp[j].value) count += dp[j].count;
            if (value == dp[j-v].value + w) count += dp[j-v].count;
            count %= mod;
            dp[j] = {value, count};
        }
    }
    cout << dp[V].count % mod << endl;
    return 0;
}
```

#### 求具体背包问题的方案
这个问题有很多类似的变种：  
- 求具体的某一个方案，方案不限
- 求所有的方案 
- 求所有方案中以序号或者某一指标按字典序排序后的方案  

其实只要清楚了第二个题的解决方案，所有题都可以迎刃而解。  
由于我们要求出所有的情况，所以我们需要保存所有的状态，就不能将二维的矩阵优化成滚动数组了，我们可以通过二维数组中记录的值得情况，反推初始状态。  

在反推状态的时候，如果开始遍历的时候是从物品1 ~ n，那么反推的时候自然就是需要从n ~ 1。  

**Trick**  
当我们需要有一定的顺序优先级的时候，需要把越优先的物品放到越后面，这样可以在反推的时候优先考虑到。  

考虑一道题：  
有一组数，将这组数分成两个部分，使得这两个部分的差的绝对值最小。  

分析：  
最完美的就是两组数相等，如果不能相等，即各等于`sum/2`，那么也让其中一组尽可能的接近`sum/2`,所以其实质就是01背包，将最大容量容量限制到`sum/2`，需要将各个数尽可能的填满这个背包。  

在求出了所有dp[N][N]后，可以通过DFS来求出所有的情况。  
```cpp
void DFS(const vector<vector<int>>& dp, const vector<int>& nums, int N, int index, int vol, vector<int>& vec, set<vector<int>>& res) {
    if (dp[index][vol] == 0) {
        res.insert(vec);
        return;
    }
    //如果可以取第i件物品
    if (dp[index][vol] == dp[index+1][vol-nums[index]] + nums[index]) {
        vec.push_back(nums[index]);
        DFS(dp, nums, N, index+1, vol-nums[index], vec, res);
        vec.pop_back();
    }
    //如果可以不取第i件物品
    if (dp[index][vol] == dp[index+1][vol]) {
        DFS(dp, nums, N, index+1, vol, vec, res);
    }
}
```




#### 有依赖的背包  
有些复杂，待完成
