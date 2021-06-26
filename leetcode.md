#### leetcode题解
##### 两数之和(1)
- 暴力迭代
```javascript
/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number[]}
 */
var twoSum = function(nums, target) {
    let res = new Array();
    for (let i = 0; i < nums.length - 1; i++) {
        for (j = i + 1; j < nums.length; j++) {
            if (nums[i] + nums[j] == target) {
                res.push(i);
                res.push(j);
                return res;
            }
        }
    }
    return res;
};
```
- 哈希表
```javascript
var twoSum = function(nums, target) {
    let map = new Map();
    let res = new Array();
    for (let i = 0; i < nums.length; i++) {
        map.set(nums[i], i);
    }
    for (let i = 0; i < nums.length; i++) {
        let temp = target - nums[i];
        if (map.get(temp) && map.get(temp) != i) {
            res.push(i);
            res.push(map.get(temp));
            return res;
        }
    }
    return res;
};
```

##### 两数相加(2)
- 双指针
```javascript
var addTwoNumbers = function(l1, l2) {
    let resNode = new ListNode();
    let head = resNode;
    // 是否需要进位
    let flag = false;
    while (l1 != null && l2 != null) {
        resNode.next = new ListNode();
        resNode = resNode.next;
        let currentSum = l1.val + l2.val;
        if (flag) {
            currentSum += 1;
            flag = false;
        }
        if (currentSum >= 10) {
            resNode.val = currentSum % 10;
            flag = true;
        } else {
            resNode.val = currentSum;
        }
        l1 = l1.next;
        l2 = l2.next;
    }
    while (l1 != null) {
        resNode.next = new ListNode();
        resNode = resNode.next;
        let currentSum = l1.val;
        if (flag) {
            currentSum += 1;
            flag = false;
        }
        if (currentSum >= 10) {
            resNode.val = currentSum % 10;
            flag = true;
        } else {
            resNode.val = currentSum;
        }
        l1 = l1.next;
    }
    while (l2 != null) {
        resNode.next = new ListNode();
        resNode = resNode.next;
        let currentSum = l2.val;
        if (flag) {
            currentSum += 1;
            flag = false;
        }
        if (currentSum >= 10) {
            resNode.val = currentSum % 10;
            flag = true;
        } else {
            resNode.val = currentSum;
        }
        l2 = l2.next;
    }
    if (flag) {
        resNode.next = new ListNode(1);
    }
    return head.next;
};****
```

##### 无重复字符的最长子串(3)
- 滑动窗口/双指针
```javascript
/*
    set:一个集合，存储当前窗口长度内已有的字符
    每一次将右指针向后加1，并查找集合内是否右有指针指向的字符；
    若有删除左指针指向的字符并递增左指针直到集合中不包含右指针指向的字符
*/
var lengthOfLongestSubstring = function(s) {
    if (s.length == 0) {
        return 0;
    }

    let res = 1;
    let set = new Set();
    let left = 0;
    let right = 1;
    set.add(s[0]);
    while (right < s.length) {
        if (set.has(s[right])) {
            // 之前的集合中有该元素
            while (set.has(s[right]) && left <= right) {
                set.delete(s[left]);
                left++;
            }
        } 
        // 添加当前的末尾元素
        set.add(s[right]);
        res = Math.max(res, right - left + 1);
        right++;
    }
    return res;
};
```

##### 寻找两个正序数组的中位数(4)
- 双指针遍历两个数组
```javascript
/**
 * @param {number[]} nums1
 * @param {number[]} nums2
 * @return {number}
 */
var findMedianSortedArrays = function(nums1, nums2) {
    let mergeNums = new Array();
    let count1 = 0;
    let count2 = 0;
    while (count1 < nums1.length && count2 < nums2.length) {
        if (nums1[count1] <= nums2[count2]) {
            mergeNums.push(nums1[count1]);
            count1++;
        } else {
            mergeNums.push(nums2[count2]);
            count2++;
        }
    }
    if (count1 < nums1.length) {
        while(count1 < nums1.length) {
            mergeNums.push(nums1[count1]);
            count1++;
        }
    } 
    if (count2 < nums2.length) {
        while (count2 < nums2.length) {
            mergeNums.push(nums2[count2]);
            count2++;
        }
    }
    // console.log(mergeNums);
    if (mergeNums.length % 2 == 0) {
        let mid = parseInt(mergeNums.length / 2);
        return (mergeNums[mid - 1] + mergeNums[mid]) / 2;
    } else {
        let mid = Math.floor(mergeNums.length / 2);
        return parseFloat(mergeNums[mid]);
    }
};
```
- 二分查找
```javascript
/* 假设两个数组总长度为K，中位数就是第k(k = K/2 或 k/2 + 1)小的
数。每一次查找数组A和数组B的前k/2个元素。选取其中较小的那个，从对应
数组中删除(后移数组开头位置的坐标)。若A[k/2] < B[k/2]，除去A的前
k/2个元素，在寻找在剩余的数中第k - k/2小的元素。每回减少相对于现在
一般的元素，就实现了二分 */
function find_kth_num(k, nums1, nums1_start, nums2, nums2_start) {
    //递归返回条件
    if (nums1_start >= nums1.length) {
        return nums2[nums2_start + k - 1];
    }
    if (nums2_start >= nums2.length) {
        return nums1[nums1_start + k - 1];
    }
    if (k == 1) {
        // console.log(`n1_start = ${nums1_start}`);
        // console.log(`n2_start = ${nums2_start}`);
        // console.log(`n1_val = ${nums1[nums1_start]}`);
        // console.log(`n2_val = ${nums2[nums2_start]}`);
        return nums1[nums1_start] < nums2[nums2_start] ? nums1[nums1_start] : nums2[nums2_start];
    }

    let k_local = Math.floor(k / 2);
    let nums1_mid = Math.min(nums1.length - 1, nums1_start + k_local - 1);
    let nums2_mid = Math.min(nums2.length - 1, nums2_start + k_local - 1);
    if (nums1[nums1_mid] <= nums2[nums2_mid]) {
        k = k - (nums1_mid + 1 - nums1_start);
        nums1_start = nums1_mid + 1;
        // console.log(`nums1_start = ${nums1_start}`);
        // console.log(k);
        return find_kth_num(k, nums1, nums1_start, nums2, nums2_start);
    } else {
        k = k - (nums2_mid + 1 - nums2_start);
        nums2_start = nums2_mid + 1;
        // console.log(`nums2_start = ${nums2_start}`);
        // console.log(k);
        return find_kth_num(k, nums1, nums1_start, nums2, nums2_start);
    }
}


var findMedianSortedArrays = function(nums1, nums2) {
    // nums1 = [1, 2];
    // nums2 = [1,2,3,4,5,6,7,8,9,10];
    let total_length = nums1.length + nums2.length;
    if (total_length % 2 == 0) {
        let n1 = find_kth_num(parseInt(total_length / 2), nums1, 0, nums2, 0);
        let n2 = find_kth_num(parseInt(total_length / 2) + 1, nums1, 0, nums2, 0);
        return (n1 + n2) / 2;
    } else {
        let res = find_kth_num(Math.ceil(total_length / 2), nums1, 0, nums2, 0);
        return parseFloat(res);
    }
    // find_kth_num(7, nums1, 0, nums2, 0);
    // find_kth_num(8, nums1, 0, nums2, 0);
    // console.log(n1);
    // console.log(n2);
};
```

##### 最长回文子串(5)
- 动态规划
```javascript
var longestPalindrome = function(s) {
    //dp
    let maxLength = 1;
    let begin = 0;;
    const dp = new Array(s.length).fill(0).map(i=>new Array(s.length).fill(false));

    for (let i = 0; i < s.length; i++) {
        dp[i][i] = true;
    }

    for (let currentLength = 2; currentLength <= s.length; currentLength++) {
        for (let i = 0; i < s.length; i++) {
            let j = i + currentLength - 1;
            if (j >= s.length) {
                break;
            }
            if (s[i] == s[j]) {
                // 'aa' 或者只有一个字母时其两边的字母相等也直接返回true
                if (j - i < 3) {
                    dp[i][j] = true;
                    begin = maxLength < j - i + 1 ? i : begin;
                    maxLength = Math.max(maxLength, j - i + 1);
                } else {
                    dp[i][j] = dp[i+1][j-1];
                    maxLength = dp[i+1][j-1] ? Math.max(maxLength, j - i + 1) : maxLength;
                    begin = dp[i+1][j-1] ? i : begin;
                }
            }
        }
    }
    // console.log(begin);
    // console.log(maxLength);
    return s.substring(begin, begin + maxLength);
};
```

##### 三数之和(15)
- 排序+双指针
```javascript
var threeSum = function(nums) {
    // nums = [-1,0,1,2,-1,-4,-2,-3,3,0,4]
    let res = new Array();
    if (nums.length < 3) {
       return res;
    }
    nums.sort((a, b) => a - b);
    // console.log(nums);
    for (let i = 0; i < nums.length; i++) {
        if (nums[i] > 0) {
            //因为已经排序好，若第i位置的数大于0，后面都为整数，不存在三数之和为0
            return res;
        }
        if (i > 0 && nums[i - 1] == nums[i]) {
            //跳过重复元素
            continue;
        }
        let target = 0 - nums[i];
        let left = i + 1;
        let right = nums.length - 1;
        while (left < right) {
            let tempSum = nums[left] + nums[right];
            if (tempSum < target) {
                left++;
            } else if (tempSum > target) {
                right--;
            } else {
                let tempArray = [nums[i], nums[left], nums[right]];
                res.push(tempArray);
                left++;
                right--;
                while (left < right && nums[left] == nums[left - 1] && nums[right] == nums[right + 1]) {
                    left++;
                    right--;
                }
            }
        }
    }
    return res;
};
```

##### 四数之和(18)
- 排序 + 双指针
```javascript
/*
    该题目解法与三数之和基本相同，只是增加了一层循环
*/
var fourSum = function(nums, target) {
    let res = [];
    if (nums.length < 4) {
        // 元素数量小于答案要求
        return res;
    }
    nums.sort((a, b) => a - b);
    for (let i = 0; i < nums.length - 3; i++) {
        if (i > 0 && nums[i] == nums[i - 1]) {
            // 去除以某一元素开头的重复元素
            continue;
        }
        let currentTargetI = target - nums[i];
        for (let j = i + 1; j < nums.length - 2; j++) {
            if (j > i + 1 && nums[j] == nums[j - 1]) {
                // 第二个元素应该取排序后最右边的,即第一个与下一个元素不同的元素
                continue;
            }
            let currentTargetJ = currentTargetI - nums[j];
            let left = j + 1;
            let right = nums.length - 1;
            while (left < right) {
                if (nums[left] + nums[right] > currentTargetJ) {
                    right--;
                } else if (nums[left] + nums[right] < currentTargetJ) {
                    left++;
                } else {
                    res.push([nums[i], nums[j], nums[left], nums[right]]);
                    left++;
                    right--;
                    while (left < right && nums[left] == nums[left - 1] && nums[right] == nums[right + 1]) {
                        left++;
                        right--;
                    }
                }
            }
        }
    }
    return res;
};
```

##### 合并两个有序链表(21)
```javascript
//双指针合并有序数组
var mergeTwoLists = function(l1, l2) {
    let dummyHead = new ListNode(0)
    let resNode = dummyHead;

    while (l1 != null && l2 != null) {
        if (l1.val <= l2.val) {
            resNode.next = l1;
            resNode = resNode.next;
            l1 = l1.next;
        } else {
            resNode.next = l2;
            resNode = resNode.next;
            l2 = l2.next;
        }
    }
    while (l1 != null) {
        resNode.next = l1;
        resNode = resNode.next;
        l1 = l1.next;
    }
    while (l2 != null) {
        resNode.next = l2;
        resNode = resNode.next;
        l2 = l2.next;
    }
    return dummyHead.next;
};
```

##### 合并k个升序链表(23)
- 堆排序：先取出所有链表的所有元素放入一个数组中，然后对数组用堆排序，再将排序好的数组元素一次做成节点连接成链表
```javascript
function exchangeVal(array, i, j) {
    let temp = array[i];
    array[i] = array[j];
    array[j] = temp;
}

function heapify(array, pos, size) {
     //从上到下
     //size是array的长度
    let leftChild = 2 * pos + 1;
    let rightChild = leftChild + 1;
    let largeChild = pos;

    if (leftChild < size && array[largeChild] < array[leftChild]) {
      largeChild = leftChild
    } 
    if (rightChild < size && array[largeChild] < array[rightChild]) {
      largeChild = rightChild;
    }
    if (largeChild != pos) {
      exchangeVal(array, pos, largeChild);
      heapify(array, largeChild, size);
    }
}

function buildHeap(array) {
    for (let i = Math.ceil((array.length - 1) / 2 - 1); i>= 0; i--) {
        heapify(array, i, array.length);
    }
}

function heapSort(array, size) {
    //array是已经堆化的数组
    //size是未排序的堆元素的长度
    // if (size <= 0) {
    //   return array;
    // }
    for (let i = size - 1; i >= 1; i--) {
        exchangeVal(array, 0, i);
        heapify(array, 0, i);
    }
}

var sortArray = function(nums) {
    // nums = [4, 6, 8, 5, 9, 10, 22];
    buildHeap(nums);
    heapSort(nums, nums.length);
    return nums; 
};


var mergeKLists = function(lists) {
    let tempArray = new Array();
    let resList = new ListNode(0, null);
    let dummyHead = resList;

    for (let list of lists) {
        while (list != null) {
            tempArray.push(list.val);
            list = list.next;
        }
    }

    sortArray(tempArray);
    for (let val of tempArray) {
        const node = new ListNode(val, null);
        resList.next = node;
        resList = resList.next;
    }

    return dummyHead.next;
};
```

##### 最长有效括号(32)
- 栈
```javascript
var longestValidParentheses = function(s) {
    /* 
        未解决边界条件，在stack中插入-1
        遇见(，往stack中插入该左括号的坐标，遇见)时，弹出stack的顶元素，代表匹配当前)的第一个(坐标
        当前)坐标减去上一个未被匹配的(的坐标，就是当前的最长有效括号的长度
    */
    let length = 0;
    let stack = new Array();
    stack.push(-1);
    for (let i = 0; i < s.length; i++) {
        if (s[i] == '(') {
            stack.push(i);
        } else {
            stack.pop();
            if (stack.length) {
                // stack不为空
                length = Math.max(length, i - stack[stack.length - 1]);
            } else {
                // stack为空
                stack.push(i);
            }
        }
    }
    return length;
};
```

##### 搜索旋转排序数组(33)
- 二分查找
```javascript
var search = function(nums, target) {
    if (nums.length == 0) {
        return -1;
    }
    if (nums.length == 1) {
        return nums[0] == target ? 0 : -1;
    }
    let left = 0;
    let right = nums.length - 1;
    while (left <= right) {
        let mid = parseInt((left + right) / 2);
        if (nums[mid] == target) {
            return mid;
        }
        if (nums[left] <= nums[mid]) {
            //mid左面部分是有序数组
            if (target >= nums[left] && target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {
            //mid右边区间是有序的
            if (target > nums[mid] && target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    return -1;
};
```

##### 缺失的第一个正数(41)
```javascript
var firstMissingPositive = function(nums) {
    //时间复杂度O(n)靠哈希算法
    //空间常数级别靠改变已有数组
    for (let i = 0; i < nums.length; i++) {
        if (nums[i] <= 0) {
            nums[i] = nums.length + 1;
        }
    }
    // console.log(nums);
    for (let ele of nums) {
        ele = Math.abs(ele);
        if (ele >= 1 && ele <= nums.length && nums[ele - 1] > 0) {
            nums[ele - 1] = -nums[ele - 1];
        }
    }
    // console.log(nums);
    for (let i = 0; i < nums.length; i++) {
        if (nums[i] > 0) {
            return i + 1;
        }
    }
    // console.log(nums);
    return nums.length + 1;
};
```

##### 跳跃游戏 II(45)
- 贪心算法
```javascript
/*
    每回在能跳跃的范围内寻找下一次能到达最远未知的坐标(nums[i] + i)
*/
var jump = function(nums) {
    let length = nums.length;
    let end = 0;
    let maxPosition = 0; 
    let steps = 0;
    for (let i = 0; i < length - 1; i++) {
        maxPosition = Math.max(maxPosition, i + nums[i]); 
        if (i == end) {
            end = maxPosition;
            steps++;
        }
    }
    return steps;
}
```

##### 全排列(46)
- 回溯算法(dfs/递归)
```javascript
function dfs(nums, currentArray, used, res) {
    if (currentArray.length == nums.length) {
        /*
            res.push(new Array(...currentArray))执行深复制时，
            当currentArray只有一个元素时，new Array()会把结构后的
            值当作新数组的元素个数。当只有一个元素时程序执行结果不正确
         */
        res.push([...currentArray]);
        return;
    }
    for (let i = 0; i < nums.length; i++) {
        if (!used[i]) {
            currentArray.push(nums[i]);
            used[i] = true;
            dfs(nums, currentArray, used, res);
            used[i] = false;
            currentArray.pop();
        }
    }
}

var permute = function(nums) {
    // 回溯算法(深度优先搜索/递归)
    // 标记nums数组中哪一为用过
    let used = new Array(nums.length).fill(false);
    let res = [];
    dfs(nums, [], used, res);
    return res;
}; 
```

##### 全排列II(47)
- dfs
```javascript
function dfs(array, map, length, res) {
    if (array.length == length) {
        res.push([...array]);
        return;
    }
    for (let key of map.keys()) {
        if (map.get(key) !== 0) {
            let value = map.get(key);
            array.push(key);
            map.set(key, --value);
            dfs(array, map, length, res);
            map.set(key, ++value);
            array.pop();
        }
    }
}

var permuteUnique = function(nums) {
    let res = [];
    // 存放nums中各个数字出现的次数
    let map = new Map();
    for (let ele of nums) {
        if (map.has(ele)) {
            map.set(ele, map.get(ele) + 1);
        } else {
            map.set(ele, 1);
        }
    }
    dfs([], map, nums.length, res);
    return res;
};
```

##### 最大子序和(53)
- 动态规划
```javascript
var maxSubArray = function(nums) {
    /* 若前i-1项的最大和小于0，不管nums[i]是否大于0，都应该取nums[i-1]得到前i项最大和；若前i-1项大于零，前i项的最大和应该加上num[i]，不管nums[i]是否大于零 */
    let sum = nums[0];
    let max = sum;
    for (let i = 1; i < nums.length; i++) {
        sum = Math.max(sum + nums[i], nums[i]);
        max = Math.max(max, sum);
    }
    return max;
};
```

##### 合并区间(56)
- 排序
``` javascript
var merge = function(intervals) {
    // 排序区间内元素，先按第一个元素从小到大，再按第二个元素从小到大
    intervals.sort((a1, a2) => {
        if (a1[0] < a2[0]) {
            return -1;
        } else if (a1[0] > a2[0]) {
            return 1;
        } else {
            return a1[1] - a2[1];
        }
    });
    let uniqIntervals = intervals.filter((item, index, array) => {
        if (index == array.length - 1) {
            return item;
        } else if (item[0] != array[index + 1][0]) {
            return item;
        }
    });
    let res = [uniqIntervals[0]];
    for (let i = 1; i < uniqIntervals.length; i++) {
        let currentInterval = res[res.length - 1];
        if (currentInterval[1] >= uniqIntervals[i][0]) {
            // 前一个区间的上限大于等于后一个区间的下限，合并区间
            // 且需要更新res中最后一个元素,为两个区间中上限较大的那个
            currentInterval[1] = Math.max(uniqIntervals[i][1], currentInterval[1]);
        } else {
            // 两个区间不用合并
            res.push(uniqIntervals[i]);
        }
    }
    return res;
};
```

##### 爬楼梯(70)
- 构建二叉树，中序遍历
```javascript
function TreeNode(val, left, right) {
  this.val = (val===undefined ? 0 : val)
  this.left = (left===undefined ? null : left)
  this.right = (right===undefined ? null : right)
}

var climbStairs = function(n) {
  //中序遍历
  //结果正确，但超时
  let root = new TreeNode(0);
  let res = 0;
  let stack = new Array();
  let currentNode = root;

  while (stack.length || currentNode) {
      while (currentNode) {
          if (currentNode.val == n) {
              res++;
              break;
          } else if (currentNode.val > n) {
              break;
          } else {
              stack.push(currentNode);
              currentNode.left = new TreeNode(currentNode.val + 1);
              currentNode = currentNode.left;
          }
      }
      currentNode = stack.pop();
      if (currentNode) {
          currentNode.right = new TreeNode(currentNode.val + 2);
          currentNode = currentNode.right;
      }
  }

  return res;
};
```
- 动态规划
```javascript
var climbStairs = function(n) {
    let dp = new Array();
    dp[0] = 1;
    dp[1] = 1;
    for (let i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
};
```

##### 对称二叉树(101)
- 递归
```javascript
/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {TreeNode} root
 * @return {boolean}
 */
function compare(left_child, right_child) {
    if (!left_child && !right_child) {
        return true;
    }
    if (!left_child || !right_child) {
        return false;;
    }
    if (left_child.val != right_child.val) {
        return false;
    } 
    return compare(left_child.left, right_child.right) && compare(left_child.right, right_child.left);
}

var isSymmetric = function(root) {
    if (!root) {
        return false;
    }
    return compare(root.left, root.right);
};
```
- 迭代
```javascript
/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {TreeNode} root
 * @return {boolean}
 */

var isSymmetric = function(root) {
    if (!root) {
        return false;
    } else if (!root.left && !root.right) {
        return true;
    } else if (!root.left || !root.right) {
        return false;
    }
    let queue = new Array();
    queue.push(root.left, root.right);
    while (queue.length) {
        let firstNode = queue.shift();
        let secondNode = queue.shift();
        if (!firstNode && !secondNode) {
            continue;
        }
        if (!firstNode || !secondNode) {
            return false;
        }
        if (firstNode.val != secondNode.val) {
            return false;
        }
        queue.push(firstNode.left);
        queue.push(secondNode.right);
        queue.push(firstNode.right);
        queue.push(secondNode.left);
    }
    return true;
};
```

##### 二叉树层序遍历(102)
```javascript
/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {TreeNode} root
 * @return {number[][]}
 */
var levelOrder = function(root) {
    let res = new Array();
    let queue = new Array();
    if (root) {
        queue.push(root);
    }

    while (queue.length) {
        let levelArray = new Array();
        let levelLength = queue.length;
        while (levelLength) {
            const node = queue.shift();
            levelArray.push(node.val);
            if (node.left) {
                queue.push(node.left);
            }
            if (node.right) {
                queue.push(node.right);
            }
            levelLength--;
        }
        res.push(levelArray);
    }
    return res;
};
```

##### 二叉树最大深度
- 递归解法(自顶向下)
```javascript
/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {TreeNode} root
 * @return {number}
 */
function findMaxDepth(node, depth) {
    if (!node) {
        return depth;
    } else {
        depth++;
        let leftRes = findMaxDepth(node.left, depth);
        let rightRes = findMaxDepth(node.right, depth);
        return leftRes >= rightRes ? leftRes : rightRes;
    } 
}

var maxDepth = function(root) {
    let res = findMaxDepth(root, 0);
    return res;
};
```

##### 二叉树的cf遍历(144)
- 迭代解法
```javascript
/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {TreeNode} root
 * @return {number[]}
 */
var preorderTraversal = function(root) {
    let res = new Array();
    let stack = new Array();
    let currentNode;

    if (root) {
        stack.push(root);
    }

    while (stack.length) {
        currentNode = stack.pop();
        res.push(currentNode.val);

        //因为是栈结构，先进右再进左
        if (currentNode.right) {
            stack.push(currentNode.right);
        }
        if (currentNode.left) {
            stack.push(currentNode.left);
        }
    }
    return res;
};
```
- 递归解法
```javascript
function pre (node, res) {
    if (node) {
        res.push(node.val);
        pre(node.left, res);
        pre(node.right, res);
    }
}

var preorderTraversal = function(root) {
    let res = new Array();
    
    pre(root, res);

    return res;
};
```

##### 二叉树的中序遍历(94)
- 迭代
```javascript
/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {TreeNode} root
 * @return {number[]}
 */

var inorderTraversal = function(root) {
    let res = new Array();
    let stack = new Array();
    let currentNode = root;

    while (stack.length || currentNode) {
        while (currentNode) {
            stack.push(currentNode);
            currentNode = currentNode.left;
        }
        currentNode = stack.pop();
        res.push(currentNode.val);
        currentNode = currentNode.right;
    }
    return res;
};
```
- 递归
```javascript
function inorder(node, res) {
    if (node) {
        inorder(node.left, res);
        res.push(node.val);
        inorder(node.right, res);
    }
    return;
}

var inorderTraversal = function(root) {
    let res = new Array();
    inorder(root, res);
    return res;
};
```

##### 二叉树的后序遍历(145)
```javascript
- 递归
/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {TreeNode} root
 * @return {number[]}
 */
function postorder(node, res) {
    if (node) {
        if (node.left) {
            postorder(node.left, res);
        } 
        if (node.right) {
            postorder(node.right, res);
        } 
        res.push(node.val);
    }
    return;
}

var postorderTraversal = function(root) {
    let res = new Array();
    postorder(root, res);
    return res;
};
```
- 迭代
```javascript
var postorderTraversal = function(root) {
    let res = new Array();
    let dequeue = new Array();

    if (root) {
        dequeue.push(root);
    }
    /** 因为后序遍历顺序是左右根，但按此顺序执行较困难，所以可以按照根右左的顺序添加答案 **/
    while (dequeue.length) {
        const node = dequeue.pop();
        res.unshift(node.val);
        if (node.left) {
            dequeue.push(node.left)
        }
        if (node.right) {
            dequeue.push(node.right);
        }
    }
    return res;
};
```

##### 从前序与中序遍历序列构造二叉树(105)
- 递归
```javascript
var buildTree = function(preorder, inorder) {
    let inorderMap = new Map();
    inorder.forEach((item, index) => {
        inorderMap.set(item, index);
    })

    function fn(start, end) {
        if (start > end) {
            return null;
        }
        let rootVal = preorder.shift();
        let root = new TreeNode(rootVal);
        let rootIndex = inorderMap.get(rootVal);
        root.left = fn(start, rootIndex - 1);
        root.right = fn(rootIndex + 1, end);
        return root;
    }
    let res = fn(0, inorder.length - 1);
    return res;
};
```

##### 从中序与后序遍历序列构造二叉树(106)
- 递归
```javascript
var buildTree = function(inorder, postorder) {
    let inorderMap = new Map();
    //对中序遍历建立哈希索引，只需时间复杂度O(1)即可查找到元素
    inorder.forEach((item, index) => {
        inorderMap.set(item, index);
    })

    function fn(start, end) {
        if (start > end) {
            return null;
        }
        let rootVal = postorder.pop();
        let root = new TreeNode(rootVal);
        let rootIndex = inorderMap.get(rootVal);
        root.right = fn(rootIndex + 1, end);
        root.left = fn(start, rootIndex - 1);
        return root;
    }
    let res = fn(0, inorder.length - 1);
    return res;
};
```

##### 路径总和(112)
- 递归
```javascript
var hasPathSum = function(root, targetSum) {
    if (root == null) {
        return false;
    }                // 遍历到null节点
  if (root.left == null && root.right == null) { // 遍历到叶子节点
    return targetSum - root.val == 0;                  // 如果满足这个就返回true。否则返回false
  }
  // 当前递归问题 拆解成 两个子树的问题，其中一个true了就行
  return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum - root.val);
};
```
- BFS
```javascript
var hasPathSum = function(root, targetSum) {
    if (!root) {
        return false;
    }
    let nodeQueue = new Array();
    let sumQueue = new Array();
    let currentNode = root;
    let currentSum = 0;
    nodeQueue.push(root);
    sumQueue.push(root.val);
    let flag = false;

    while (nodeQueue.length) {
        currentNode = nodeQueue.shift();
        currentSum = sumQueue.shift();
        if (currentNode.left) {
            nodeQueue.push(currentNode.left);
            sumQueue.push(currentSum + currentNode.left.val);
        }
        if (currentNode.right) {
            nodeQueue.push(currentNode.right);
            sumQueue.push(currentSum + currentNode.right.val);
        }
        if (!currentNode.left && !currentNode.right && currentSum == targetSum) {
            flag = true;
            break;
        }
    }
    return flag;
};
```

##### 填充每个节点的下一个右侧节点指针(116)
- BFS
```javascript
var connect = function(root) {
    if (!root) {
        return root;
    }
    let queue = new Array();
    let currentNode;
    let currentLength;

    queue.push(root);
    while (queue.length) {
        currentLength = queue.length;
        for (let i = 0; i < currentLength; i++) {
            currentNode = queue.shift();
            if (i == currentLength - 1) {
                currentNode.next = null;
            } else {
                currentNode.next = queue[0];
            }
            if (currentNode.left || currentNode.right) {
                queue.push(currentNode.left);
                queue.push(currentNode.right);
            }
        }
    }
    return root;
};
```

##### 填充每个节点的下一个右侧节点指针 II(117)
```javascript
var connect = function(root) {
    if (!root) {
        return root;
    }
    let queue = new Array();
    let currentNode;
    let currentLength;

    queue.push(root);
    while (queue.length) {
        currentLength = queue.length;
        for (let i = 0; i < currentLength; i++) {
            currentNode = queue.shift();
            if (i == currentLength - 1) {
                currentNode.next = null;
            } else {
                currentNode.next = queue[0];
            }
            if (currentNode.left) {
                queue.push(currentNode.left);
            }
            if (currentNode.right) {
                queue.push(currentNode.right);
            }
        }
    }
    return root;
};
```

##### 买卖股票的最佳时机(121)
- 动态规划
```javascript
/*
    1. 动态规划做题步骤
        明确 dp(i) 应该表示什么（二维情况：dp(i)(j)）；
        根据 dp(i) 和 dp(i−1) 的关系得出状态转移方程；
        确定初始条件，如 dp(0)。
*/
var maxProfit = function(prices) {
    // dp[i]当前天数的最大利润
    // dp[i] = max(dp[i - 1], prices[i] - minprice)
    let minPrice = prices[0];
    let profit = -Infinity;
    for (let i = 1; i < prices.length; i++) {
        let currentProfit = prices[i] - minPrice;
        if (currentProfit > profit) {
            profit = currentProfit;
        }
        if (minPrice > prices[i]) {
            minPrice = prices[i];
        }
    }
    return profit >= 0 ? profit : 0;
};
```

##### 买卖股票的最佳时机II(122)
- 贪心算法
```javascript
/*
    只要前一天价格比后一天价格低，就能获益，所以应该累加获益
*/
var maxProfit = function(prices) {
    // 贪心算法
    let profit = 0;
    for (let i = 0; i < prices.length - 1; i++) {
        if (prices[i] <= prices[i + 1]) {
            profit += prices[i + 1] - prices[i];
        }
    }
    return profit;
};
```

##### 买卖股票的最佳时机III(123)
```javascript
/*
    只进行过一次买操作；
    进行了一次买操作和一次卖操作，即完成了一笔交易；
    在完成了一笔交易的前提下，进行了第二次买操作；
    完成了全部两笔交易。
*/
var maxProfit = function(prices) {
    let buy1 = -prices[0];
    let sell1 = 0;
    let buy2 = -prices[0];
    let sell2 = 0;
    for (let i = 1; i < prices.length; i++) {
        buy1 = Math.max(buy1, -prices[i]);
        sell1 = Math.max(sell1, prices[i] + buy1);
        buy2 = Math.max(buy2, sell1 - prices[i]);
        sell2 = Math.max(sell2, prices[i] + buy2);
    }
    return Math.max(sell1, sell2);
};
```

##### 对链表进行插入排序(147)
  ```javascript
  var insertionSortList = function(head) {
    if (head == null) {
        return head;
    }
    //创建一个假头部，便于在头部之前插入元素
    let dummyHead = new ListNode(0, head);
    //记录已经排序好的链表的最后一位，从后面lastSorted
    //后查找需要插入的元素
    let lastSorted = head;
    let curr = head.next;

    while (curr) {
        if (lastSorted.val <= curr.val) {
            lastSorted = lastSorted.next;
        } else {
            let prev = dummyHead;
            while (prev.next) {
                if (prev.next.val > curr.val) {
                    lastSorted.next = curr.next;
                    curr.next = prev.next;
                    prev.next = curr;
                    //curr前的node都已经排序完成，只需要插入一次即可
                    break;
                } else {
                    prev = prev.next;
                }
            }
        }
        curr = lastSorted.next;
    }

    return dummyHead.next;
  };
  ```

##### 乘积最大子数组(152)
- 动态规划
```javascript
var maxProduct = function(nums) {
    /*
        需要两个值max和min，max代表以nums[i]结尾的乘积最大子数组
        的乘积，min代表以nums[i]结尾的乘积最小的子数组的乘积
    */
    let max = nums[0];
    let min = nums[0];
    let res = max;
    for (let i = 1; i < nums.length; i++) {
        let currentMax = max;
        let currentMin = min;
        max = Math.max(nums[i], currentMax * nums[i], currentMin * nums[i]);
        min = Math.min(nums[i], currentMax * nums[i], currentMin * nums[i]);
        res = Math.max(res, max);
    }
    return res;
};
```

##### 打家劫舍(198)
- 动态规划
```javascript
var rob = function(nums) {
    /* 
    对于第n位的数字来说dp[n] = MAX(tempMax, dp[n - 2] + nums[n])
    下一轮的dp[n-2]是上一轮的tempMax
    */
    if (nums.length <= 1) {
        return nums[0];
    }
    // tempMax:截止到当前位前一位时，可以获得的最大金额
    // dp:截止到当前位前两位时， 可以获得的最大金额
    let tempMax = Math.max(nums[0], nums[1]);
    let dp = nums[0];
    for (let i = 2; i < nums.length; i++) {
        let temp = tempMax;
        tempMax = Math.max(tempMax, dp + nums[i]);
        dp = temp;
    }
    return tempMax;
};
```

- 动态规划
```javascript
/*
注意到若{nums}nums 中不存在某个元素 x，则选择任一小于
x 的元素不会影响到大于 x 的元素的选择。因此我们可以将nums排
序后，将其划分成若干连续子数组，子数组内任意相邻元素之差不超过1。对
每个子数组按照方法一的动态规划过程计算出结果，累加所有结果即为答案。
*/
function compare(v1, v2) {
    if (v1 < v2) {
        return -1;
    } else if (v1 > v2) {
        return 1;
    } else {
        return 0;
    }
}

function rob(nums) {
    if (nums.length == 1) {
        return nums[0];
    }
    let tempMax = Math.max(nums[0], nums[1]);
    let dp = nums[0];
    for (let i = 2; i < nums.length; i++) {
        let temp = tempMax;
        tempMax = Math.max(tempMax, dp + nums[i]);
        dp = temp;
    }
    return tempMax;
}

var deleteAndEarn = function(nums) {
    nums.sort(compare);
    let subNums = [nums[0]];
    let res = 0;
    for (let i = 1; i < nums.length; i++) {
        // console.log(subNums);
        if (nums[i] == nums[i - 1]) {
            // 当前元素与前一个元素相等，将它们的累加
            subNums[subNums.length - 1] += nums[i];
        } else if (nums[i] == nums[i - 1] + 1) {
            // 当前元素比前一个元素大1，将当前元素添加到数组中
            subNums.push(nums[i]);
        } else {
            // 获得一个基于nums的子数组，数组中的元素是元素的和
            // 相邻元素是互相之间差值为1
            res += rob(subNums);
            subNums = [nums[i]];
        }
    }
    res += rob(subNums);
    return res;
};
```

##### 岛屿数量(200)
  - BFS
    ```javascript
    function bfs(grid, i, j) {
      const i_limit = grid.length;
      const j_limit = grid[0].length;
      const DIRECTION = [[-1, 0], [1, 0], [0, -1], [0, 1]];
      let queue = new Array();
      grid[i][j] = 0;
      queue.push([i, j]);

      while (queue.length != 0) {
        const current = queue.shift();
        for (let dir of DIRECTION) {
            const row = current[0] + dir[0];
            const column = current[1] + dir[1];
            if (row < 0 || row >= i_limit || column < 0 || column >= j_limit || grid[row][column] == 0) {
                continue;
            }
            queue.push([row, column]);
            grid[row][column] = 0;
        }
      }
    }

    var numIslands = function(grid) {
      const i_limit = grid.length;
      const j_limit = grid[0].length;
      let res = 0;
      for (let i = 0; i < i_limit; i++) {
          for (let j = 0; j < j_limit; j++) {
              if (grid[i][j] == 1) {
                  res++;
                  bfs(grid, i , j);
              }
          }
      }
      return res;
    };
    ```
  - DFS
    ```javascript
    function dfs(grid, i, j) {
      const i_limit = grid.length;
      const j_limit = grid[0].length;
      const DIRECTION = [[-1, 0], [1, 0], [0, -1], [0, 1]];
      let stack = new Array();
      grid[i][j] = 0;
      stack.push([i, j]);

      while (stack.length != 0) {
        const current = stack.pop();
        for (let dir of DIRECTION) {
            const row = current[0] + dir[0];
            const column = current[1] + dir[1];
            if (row < 0 || row >= i_limit || column < 0 || column >= j_limit || grid[row][column] == 0) {
                continue;
            }
            stack.push([row, column]);
            grid[row][column] = 0;
        }
      }
    }

    var numIslands = function(grid) {
      const i_limit = grid.length;
      const j_limit = grid[0].length;
      let res = 0;

      for (let i = 0; i < i_limit; i++) {
        for (let j = 0; j < j_limit; j++) {
            if (grid[i][j] == 1) {
                res++;
                dfs(grid, i , j);
            }
        }
      }
      return res;
    };
    ```

##### 打家劫舍II(213)
- 动态规划
```javascript
function fn(nums, start, end) {
    let tempMax = Math.max(nums[start], nums[start + 1]);
    let dp = nums[start];
    for (let i = start + 2; i < end; i++) {
        let temp = tempMax;
        tempMax = Math.max(tempMax, dp + nums[i]);
        dp = temp;
    }
    return tempMax;
}

var rob = function(nums) {
    // 若偷窃第一栋房屋，不遍历最后一个房屋；若偷窃最后一个房屋，不遍历第一个房屋
    // 边界条件
    if (nums.length == 1) {
        return nums[0];
    } else if (nums.length == 2) {
        return Math.max(nums[0], nums[1]);
    } else {
        let r1 = fn(nums, 0, nums.length - 1);
        let r2 = fn(nums, 1, nums.length);
        return Math.max(r1, r2);
    }
};
```

##### 用队列实现栈(225)
  - 单队列实现栈
    ```javascript
    MyStack.prototype.push = function(x) {
      /** 每插入一个元素就逐个弹出之前的元素然后依次插入队列中 **/
      let preLength = this.queue.length;
      this.queue.push(x);
      for (let i = 0; i < preLength; i++) {
          let ele = this.queue.shift();
          this.queue.push(ele);
      }
    };

    MyStack.prototype.pop = function() {
        return this.queue.shift();
    };

    MyStack.prototype.top = function() {
        return this.queue[0];
    };

    MyStack.prototype.empty = function() {
        return !this.queue.length;
    };
    ```

##### 用栈实现队列(232)
  - 双栈实现队列
    ```javascript
    var MyQueue = function() {
      this.inputStack = new Array();
      this.outputStack = new Array();
    };

    MyQueue.prototype.push = function(x) {
      this.inputStack.push(x);
    };

    MyQueue.prototype.pop = function() {
      /** 
          若输出队列为空，则将输入队列中的全部元素
          弹出然后压入到输出队列中。若不为空则直接弹出
          栈顶元素。
      **/
      if (this.outputStack.length == 0) {
          while (this.inputStack.length != 0) {
            this.outputStack.push(this.inputStack.pop());
          }
      }
      return this.outputStack.pop();
    };

    MyQueue.prototype.peek = function() {
      if (this.outputStack.length == 0) {
          while (this.inputStack.length != 0) {
            this.outputStack.push(this.inputStack.pop());
          }
      }
      return this.outputStack[this.outputStack.length - 1];
    };

    MyQueue.prototype.empty = function() {
      if (!this.inputStack.length && !this.outputStack.length) {
          return true;
      }
      return false;
    };
    ```

##### 移动零(283)
  - 冒泡排序写法
  ```javascript
  var moveZeroes = function(nums) {
    let swap = true;
    for (let i = 0; i < nums.length - 1; i++) {
        if (!swap) {
            break;
        }
        swap = false;
        for (let j = 0; j < nums.length - i- 1; j++) {
            if (nums[j] == 0) {
                nums[j] = nums[j + 1];
                nums[j + 1] = 0;
                swap = true;
            }
        }
    }
  };
  ```

##### 二叉树的最近公共祖先(236)
- dfs
```javascript
function dfs(node, target, path) {
    //递归返回条件
    if (!node) {
        return false;
    }else if (node.val == target.val) {
        // path.push(node);
        return true;
    }
    path.push(node);
    if (dfs(node.left, target, path)) {
        return true;
    }
    if (dfs(node.right, target, path)) {
        return true;
    }
    path.pop();
    return false;
} 

function findAncestor(a1, a2, node) {
    //a1.length >= a2.length
    for (let i = a1.length - 1; i >= 0; i--) {
        if (i >= a2.length && a1[i].val == node.val) { 
            return node;
        } else if (i < a2.length && a1[i].val == a2[i].val) {
            return a1[i];
        }
    }
}

var lowestCommonAncestor = function(root, p, q) {
    let p_path = new Array();
    let q_path = new Array();
    let res;

    dfs(root, p, p_path);
    dfs(root, q, q_path);
    // console.log(p_path[0].val);
    // console.log(q_path[0].val);
    
    if (p_path.length >= q_path.length) {
        res = findAncestor(p_path, q_path, q);
    } else {
        res = findAncestor(q_path, p_path, p);
    }
    return res;
};
```
- 后序遍历
```javascript
var lowestCommonAncestor = function(root, p, q) {
    //后序遍历
    if (root == null || root == p || root == q) {
        return root;
    }
    let left = lowestCommonAncestor(root.left, p, q);
    let right = lowestCommonAncestor(root.right, p, q);
    if (!left) {
        return right;
    }
    if (!right) {
        return left;
    }
    return root;
};
```

##### 最长递增子序列(300)
- 动态规划(O(n2))
```javascript
var lengthOfLIS = function(nums) {
    let map = new Map();
    let res = 1;
    for (let i = 0; i < nums.length; i++) {
        let currentEle = nums[i];
        let currentLength = 0;
        for (let j = 0; j < i; j++) {
            if (nums[j] < nums[i]) {
                currentLength = Math.max(currentLength, map.get(j));
            }
        }
        currentLength += 1;
        map.set(i, currentLength);
        res = Math.max(res, currentLength);
    }
    return res;
};
```
- 动态规划(O(nlogn))
```javascript
function findEle(target, array, start, end) {
    //在array数组中找到比targrt大的最小元素的索引
    let pos = 0;
    while (start <= end) {
        let mid = parseInt((start + end) / 2);
        if (array[mid] == target) {
            return mid;
        } else if (array[mid] < target) {
            pos = mid;
            start = mid + 1;
            //console.log(`Less: mid=${mid}, start=${start}, end=${end}, array[mid]=${array[mid]}`);
        } else {
            end = mid - 1;
            //console.log(`More: mid=${mid}, start=${start}, end=${end}, array[mid]=${array[mid]}`);
        }
    }
    return pos + 1;
}

var lengthOfLIS = function(nums) {
    //nums = [-147,-171,-584,590,501,13,489,-938,396,-544,-229,697,157,-933];
    let array = new Array(nums.length + 1);
    let length = 1;
    array[0] = Number.MIN_SAFE_INTEGER;
    array[1] = nums[0];
    for (let i = 1; i < nums.length; i++) {
        if (nums[i] > array[length]) {
            array[length + 1] = nums[i];
            length++;
            //console.log(`push: ${array}`);
        } else {
            let pos = findEle(nums[i], array, 0, length);
            array[pos] = nums[i];
            //console.log(`find: ${array}`);
        }
    }
    return length;
};
```

##### 俄罗斯套娃信封问题(354)
- 动态规划(O(n2))
```javascript
// 是问题最长递增子序列(300)的二维化，解法相同
var maxEnvelopes = function(envelopes) {
    //envelopes = [[2,100],[3,200],[4,300],[5,500],[5,400],[5,250],[6,370],[6,360],[7,380]];
    // 排序，然后遍历，O(n2)
    envelopes = envelopes.sort((x, y) => {
        if (x[0] != y[0]) {
            return x[0] - y[0];
        } else {
            return x[1] - y[1];
        }
    })
    //console.log(envelopes);
    let resLength = 1;
    let dp = new Array(envelopes.length).fill(1);
    for (let i = 0; i < envelopes.length; i++) {
        //let currentEnvelop = envelopes[i];
        for (let j = 0; j < i; j++) {
            if (envelopes[i][0] > envelopes[j][0] && envelopes[i][1] > envelopes[j][1]) {
                //当前信封大于envelops[j]对应的信封
                if (dp[j] + 1 > dp[i]) {
                    dp[i] = dp[j] + 1;
                }
            }
        }
        resLength = Math.max(resLength, dp[i]);
    }
    return resLength;
};
```

- 动态规划(O(nlogn))
```javascript
function findPos(target, array, start, end) {
    let pos = 0;
    while (start <= end) {
        let mid = parseInt((start + end) / 2);
        if (array[mid] == target) {
            return mid;
        } else if (array[mid] < target) {
            pos = mid;
            start = mid + 1;
        } else {
            end = mid - 1;
        }
    }
    return pos + 1;
}

var maxEnvelopes = function(envelopes) {
    //envelopes = [[2,100],[3,200],[4,300],[5,500],[5,400],[5,250],[6,370],[6,360],[7,380]];
    envelopes = envelopes.sort((x, y) => {
        // 先按宽度升序排序，在按高度降序排序
        if (x[0] != y[0]) {
            return x[0] - y[0];
        } else {
            return y[1] - x[1];
        }
    })
    //console.log(envelopes);
    let res= 1;
    let height = new Array();
    height[0] = 0;
    height[1] = envelopes[0][1];
    for (let i = 0; i < envelopes.length; i++) {
        /* 
            在排序好的数组中，只要下一个元素的高度大于height的结元素
            就一定可以放下前一个信封，(按照排序规则这时下一个元素的
            宽度也必定大于结尾元素的宽度)
        */
        if (envelopes[i][1] > height[height.length - 1]) {
            res++;
            height.push(envelopes[i][1]);
        } else {
            let pos = findPos(envelopes[i][1], height, 0, height.length - 1);
            height[pos] = envelopes[i][1];
        }
    }
    return res;
};
```

##### 矩形区域不超过 K 的最大数值和(363)
- 动态规划(On2mlogm)
```javascript
/* 
    思路与最大子矩阵相同，都是二维转一维
    该题目是寻找区间和sl - si >= k
    sl:当前区间和；si:之前的区间和
    遍历sl，同时存储之前的si，在之前的
    si中找到比sl-k大的最小的那个si
*/
function compare(v1, v2) {
    if (v1 < v2) {
        return - 1;
    } else if (v1 > v2) {
        return 1;
    } else {
        return 0;
    }
}

function findEle(array, target) {
    // 在array中找到刚好比target大的元素或等于target的元素
    array.sort(compare);
    let start = 0;
    let end = array.length - 1;
    let pos = array.length;
    while (start <= end) {
        let mid = parseInt((start + end) / 2);
        if (array[mid] == target) {
            return mid;
        } else if (array[mid] < target) {
            start = mid + 1;
        } else {
            pos = mid;
            end = mid - 1;
        }
    }
    return pos;
}

var maxSumSubmatrix = function(matrix, k) {
    let M = matrix.length;
    let N = matrix[0].length;
    let res = Number.MIN_SAFE_INTEGER;
    let tempArray = new Array(N);

    for (let i = 0; i < M; i++) {
        tempArray.fill(0);
        for (let j = i; j < M; j++) {
            for (let l = 0; l < N; l++) {
                tempArray[l] += matrix[j][l];
            }
            let sums = new Array();
            sums[0] = 0;
            let tempSum = 0;
            for (let p = 0; p < N; p++) {
                tempSum += tempArray[p]
                let target = tempSum - k;
                let pos = findEle(sums, target);
                if (pos < sums.length) {
                    res = Math.max(res, tempSum - sums[pos]);
                }
                sums.push(tempSum);
                //console.log(pos);
            }
        }
    }

    return res;
};
```

##### 最大整除子集(368)
- 动态规划
```javascript
function compare(v1, v2) {
    return v1 - v2;
}

var largestDivisibleSubset = function(nums) {
    let array = new Array(nums.length).fill(1);
    let length = 1;
    // 最长整除子集的最后一个元素的索引
    let index = 0;
    // 排序数组，确保每回遍历只需要寻找前面的元素
    nums = nums.sort(compare);
    for (let i = 0; i < nums.length; i++) {
        let tempLength = 1
        for (let j = i - 1; j >= 0; j--) {
            // 向前遍历数组，寻找能被当前元素nums[i]整除的元素
            if (nums[i] % nums[j] == 0) {
                tempLength = Math.max(tempLength, array[j] + 1);
            }
        }
        array[i] = tempLength;
        if (tempLength > length) {
            length = tempLength;
            index = i;
        }
    }
    // console.log(array);
    // console.log(length);
    // 通过array回溯，找到最长整除子集
    let res = [nums[index]];
    let nextLength = length - 1;
    let currentEle = nums[index];
    for (let i = index - 1; i >= 0 && nextLength > 0; i--) {
        if (array[i] == nextLength && currentEle % nums[i] == 0) {
            res.unshift(nums[i]);
            currentEle = nums[i];
            nextLength--;
        }
    }
    //console.log(res);
    return res;
};
```

##### 字符串解码(394)
  - 栈解法
  ```javascript
  var decodeString = function(s) {
    let res = '';
    let num = 0;
    let stack = new Array();

    for (let i = 0; i < s.length; i++) {
        if (!isNaN(s[i])) {
            while (!isNaN(s[i])) {
                num = num * 10 + Number(s[i] - '0');
                i++;
            }
            stack.push(num);
            num = 0;
            i--;
        } else if (s[i] == ']') {
            let tempNum = 0;
            let seed = '';
            let tempRes = '';
            let temp = new Array();
            let tempEle = stack.pop();
            while (tempEle != '[') {
                temp.push(tempEle);
                tempEle = stack.pop();
            }
            tempNum = stack.pop();
            seed = temp.reverse().join('');
            tempRes = seed.repeat(tempNum);
            stack.push(tempRes);
        } else {
            stack.push(s[i]);
        }
    }

    return stack.join('');
  };
  ```
  - 优化解法(算法相同)
  ```javascript
  var decodeString = function(s) {
    let mulStack = [], strStack = [], num = 0, res = ''
    for (const c of s) {   
        if (!isNaN(c)) {  
            num = num * 10 + (c - '0')
        } else if (c == '[') {  
            strStack.push(res)
            mulStack.push(num) 
            res = '' 
            num = 0
        } else if (c == ']') {  
            res = strStack.pop() + res.repeat(mulStack.pop())
        } else {                   
            res += c
        }
    }
    return res;
  };
  ```

##### 目标和(494)
  - DFS
  ```javascript
  var findTargetSumWays = function(nums, S) {
    //深度优先可以考虑用压栈的方法，栈的最大深度O(h), h为图的高度，比BFS优化很多
    let res = 0;
    const dfs = (index, sum) => {
        if (index < nums.length) {
            dfs(index + 1, sum + nums[index]);
            dfs(index + 1, sum - nums[index]);
        } else if (index == nums.length && sum == S) {
            res++;
        }
        return;
    }
    dfs(0, 0);

    return res;
  };
  ```

##### 01矩阵(542)
  - BFS
  图的多源BFS算法，把所有源头加入队列，一次遍历。
  需要把图中除源头外的节点标记为未访问
  ```javascript
  var updateMatrix = function(matrix) {
    let queue = new Array();
    const r_limit = matrix.length;
    const c_limit = matrix[0].length;
    const DIRECTION = [[-1, 0], [1, 0], [0, -1], [0, 1]];

    for (let i = 0; i < r_limit; i++) {
        for (let j = 0; j < c_limit; j++) {
            if (matrix[i][j] == 0) {
                queue.push([i, j]);
            } else {
                //标记为未访问
                matrix[i][j] = -1;
            }
        }
    }

    while (queue.length) {
        const pos = queue.shift();
        for (const ele of DIRECTION) {
            const row = pos[0] + ele[0];
            const column = pos[1] + ele[1];
            if (row >= 0 && row < r_limit && column >= 0 && column < c_limit && matrix[row][column] == -1) {
                matrix[row][column] = matrix[pos[0]][pos[1]] + 1;
                queue.push([row, column]);
            }
        }
    }

    return matrix;
  };
  ```

##### 最长递增子序列(673)
```javascript
var findNumberOfLIS = function(nums) {
    // nums = [2,2,2,2,2];
    let dp = new Array(nums.length).fill(1);
    let count = new Array(nums.length).fill(1);
    let res = 1;
    for (let i = 0; i < nums.length; i++) {
        let currentEle = nums[i];
        let currentLength = 1;
        for (let j = 0; j < i; j++) {
            if (nums[j] < nums[i]) {
                //currentLength = Math.max(currentLength, dp[j]);
                if (dp[j] + 1 > dp[i]) {
                    // 说明dp[i]应该增加1
                    dp[i] = dp[j] + 1;
                    count[i] = count[j];
                } else if (dp[j] + 1 == dp[i]) {
                    count[i] += count[j];
                }   
            }
        }
        //currentLength += 1;
        //dp[i] = currentLength;
        res = Math.max(res, dp[i]);
    }
    // console.log(dp);
    // console.log(res);
    let result = 0;
    for (let i = 0; i < dp.length; i++) {
        if (dp[i] == res) {
            result += count[i];
        }
    }
    return result;
};
```

##### 图像渲染(733)
  - BFS
  ```javascript
  var floodFill = function(image, sr, sc, newColor) {
    const preColor = image[sr][sc];
    let queue = new Array();
    const r_limit = image.length;
    const c_limit = image[0].length;
    const DIRECTION = [[-1, 0], [1, 0], [0, -1], [0, 1]];

    queue.push([sr, sc]);
    while (queue.length) {
      const pos = queue.shift();
      image[pos[0]][pos[1]] = newColor;
      for (const ele of DIRECTION) {
          const row = pos[0] + ele[0];
          const column = pos[1] + ele[1];
          if (row < 0 || row >= r_limit || column < 0 || column >= c_limit || image[row][column] != preColor || image[row][column] == newColor) {
              continue;
          } else {
              queue.push([row, column]);
          }
      }
    }

    return image;
  };
  ```
  - DFS
  ```javascript
  var floodFill = function(image, sr, sc, newColor) {
    if(image[sr][sc] === newColor){
        return image;
    }
    const oldColor = image[sr][sc];
    let dfs = (sr, sc) => {
        if(sr < 0 || sr >= image.length || sc < 0 ||
        sc >= image[0].length || image[sr][sc] !== oldColor){
            return;
        }
        image[sr][sc] = newColor;
        dfs(sr + 1, sc);
        dfs(sr - 1, sc);
        dfs(sr, sc + 1);
        dfs(sr, sc - 1);
    }
    dfs(sr, sc);
    return image;
  };
  ```

##### 删除并获得点数(740)
- 动态规划
```javascript
var deleteAndEarn = function(nums) {
    let maxVal = 0;
    for (const val of nums) {
        maxVal = Math.max(maxVal, val);
    }
    const sum = new Array(maxVal + 1).fill(0);
    for (const val of nums) {
        sum[val] += val;
    }
    return rob(sum);
};

const rob = (nums) => {
    /*
        将本问题转化为打家劫舍的问题。
        对于数组nums，创建另一个数组sums。nums中相同的元素
        val，sums[val]是所有val总和，因此sums的长度是nums
        最大元素的大小。之后就可以采取打家劫舍的解法，选择了
        sums[i]就不能选择sums[i]的前一个和后一个元素
    */
    const size = nums.length;
    let first = nums[0], second = Math.max(nums[0], nums[1]);
    for (let i = 2; i < size; i++) {
        let temp = second;
        second = Math.max(first + nums[i], second);
        first = temp;
    }
    return second;
}
```

##### 使用最小花费爬楼梯(749)
- 动态规划
```javascript
/**
 * @param {number[]} cost
 * @return {number}
 */
var minCostClimbingStairs = function(cost) {
    let destination = cost.length;
    let dp = new Array(destination).fill(0);
    dp[0] = cost[0];
    dp[1] = cost[1];
    for (let i = 2; i < destination; i++) {
        dp[i] = Math.min(dp[i - 1], dp[i - 2]) + cost[i];
    }
    return Math.min(dp[destination - 1], dp[destination - 2]);
};
```

##### 使序列递增的最小交换次数(801)
- 动态规划
```javascript
var minSwap = function(A, B) {
    /* n1:不交换i-1位的情况下，前i-1位实现严格递增的交换次数
       s1:交换i-1位的情况下，前i-1位实现严格递增的交换次数 */
    let n1 = 0;
    //当A,B仅有一位时，交换A[0]和B[0]可以实现两个数组的严格递增
    let s1 = 1;
    for (let i = 1; i < A.length; i++) {
        let n2 = Number.MAX_SAFE_INTEGER;
        let s2 = Number.MAX_SAFE_INTEGER;
        if (A[i - 1] < A[i] && B[i - 1] < B[i]) {
            //在不交换前i-1位已经是严格递增数组的情况下，第i位不必交换即可实现严格递增
            n2 = n1;
            /* 在不交换前i-1位已经是严格递增数组的情况下，s2有两种可能。
               交换第i位之后，数组依然严格递增，此时s2=n1+1，下面的条件判断会判断该情况
               交换第i位之后，数组不是严格递增，此时必须交换i-1位，才能实现严格递增 */
            s2 = s1 + 1;
        }
        if (A[i - 1] < B[i] && B[i - 1] < A[i]) {
            //前一位只要交换一下就可以实现不交换i位实现数组严格递增
            n2 = Math.min(n2, s1);
            s2 = Math.min(s2, n1 + 1);
        }
        n1 = n2;
        s1 = s2;
    }
    return Math.min(n1, s1);
};
```

  ##### 钥匙和房间(841)
  - DFS
    ```javascript
    var canVisitAllRooms = function(rooms) {
      let keys = new Set();
      let stack = new Array();

      stack.push(0);
      keys.add(0);
      while(stack.length) {
        const room = stack[stack.length - 1];
        let flag = true;
        for (const ele of rooms[room]) {
            if (!keys.has(ele)) {
                stack.push(ele);
                keys.add(ele);
                flag = false;
            }
        }
        if (flag) {
            stack.pop();
        }
      }

      return keys.size == rooms.length ? true : false;
    };
    ```

##### 环形子数组的最大和(918)
- 动态规划(O(n))
```javascript
var maxSubarraySumCircular = function(nums) {
    /* 
        环形子数组的最大和分为两种情况
        1 子数组不是环形数组，直接包含在nums中
        2 子数组是环形数组，一定包含开头元素A[0]和结尾元素A[length - 1]
    */
    if (nums.length == 1) {
        return nums[0];
    }
    // 先找非环形数组最大和
    let max = nums[0];
    let res = max;
    let sum = nums[0];
    for (let i = 1; i < nums.length; i++) {
        sum += nums[i];
        max = Math.max(nums[i], max + nums[i]);
        res = Math.max(res, max);
    }
    // 找环形数组的最大和
    /* 
        若答案为环形数组，出去开头和结尾元素必须添加外，在[1, length-2]
        这个区间内必然包含一段连续和为负数的区间，导致整个数组的和sum必须
        减去该段区间的负数和
        同样的思路也可以用在寻找最大和上，只不过当答案是环形数组时，必须
        寻找中间区间的最小值
     */
    let last = nums[1];
    let min = nums[1];
    for (let i = 2; i < nums.length - 1; i++) {
        min = Math.min(nums[i], nums[i] + min);
        last = Math.min(last, min);
    }
    return Math.max(res, sum - last);
};
``` 

##### 最长的斐波那契子序列的长度(873)
- 动态规划(O(n2))
```javascript
function findEle(arr, start, end, target) {
    while (start <= end) {
        let mid = parseInt((start + end) / 2);
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            start = mid + 1;
        } else {
            end = mid - 1;
        }
    }
    return -1;
}

var lenLongestFibSubseq = function(arr) {
    let length = 0;
    for (let i = 0; i < arr.length - 2; i++) {
        let currentEle = arr[i];
        for (let j = i + 1; j < arr.length; j++) {
            let tempLength = 2;
            let currentPos = j;
            let target = arr[j] + currentEle;
            let nextPos = findEle(arr, currentPos + 1, arr.length - 1, target);
            while (nextPos != -1) {
                tempLength++;
                target = target + arr[currentPos];
                currentPos = nextPos;
                nextPos = findEle(arr, currentPos + 1, arr.length - 1, target);
            }
            if (tempLength > 2) {
            length = Math.max(length, tempLength);
        }
        }
    }
    return length;
};
```
- 动态规划(O(n2))
```javascript
/*
    改进了中间的while循环，采用map可以在O(n)的时间内找到元素
*/
var lenLongestFibSubseq = function(arr) {
    let length = 0;
    let map = new Map();

    for (let i = 0; i < arr.length; i++) {
        map.set(arr[i], i);
    }
    for (let i = 0; i < arr.length - 2; i++) {
        for (let j = i + 1; j < arr.length; j++) {
            let tempLength = 2;
            let target = arr[i] + arr[j];
            let lastPos = j;
            let currentPos = map.get(target);
            while (currentPos) {
                tempLength++;
                target = target + arr[lastPos];
                lastPos = currentPos;
                currentPos = map.get(target);
            }
            length = Math.max(tempLength, length);
        }
    }
    return length > 2 ? length : 0;
};
```

##### 最长等差数列(1027)
- 动态规划(O(n2))
```javascript
ar longestArithSeqLength = function(nums) {
    let maxEle = Math.max(...nums);
    let minEle = Math.min(...nums);
    // dp[i][j]以nums[i],nums[j]结尾的最长等差数列的长度，隐式确定了公差为nums[j]-nums[i]
    let dp = new Array(nums.length).fill(0).map(x => new Array(nums.length).fill(2))
    let map = new Map();
    let length = 2;

    for (let i = 0; i < nums.length - 1; i++) {
        for (let j = i + 1; j < nums.length; j++) {
            let diff = nums[j] - nums[i];
            let target = nums[i] - diff;
            if (map.has(target)) {
                let targetIndex = map.get(target);
                //console.log(j, diff, targetIndex, dp[targetIndex][diff], dp[j][diff]);
                dp[i][j] = dp[targetIndex][i] + 1;
                length = Math.max(length, dp[i][j]);
            }
        }
        map.set(nums[i], i);
    }
    return length;
};
```

##### 数组中重复的数字(剑指offer 3)
```javascript
/*
    因为长度为length的数组中数字不超过nums，
    此说明含义：数组元素的 索引 和 值 是 一对多 的关系。
    因此，可遍历数组并通过交换操作，使元素的 索引 与 值 一一对应（即 nums[i] = inums[i]=i ）。
    因而，就能通过索引映射对应的值，起到与字典等价的作用。
*/
var findRepeatNumber = function(nums) {
    for (let i = 0; i < nums.length; i++) {
        if (nums[i] == i) {
            continue;
        } else if (nums[nums[i]] == nums[i]) {
            return nums[i];
        } else {
            let temp = nums[i];
            nums[i] = nums[nums[i]];
            nums[temp] = temp;
        }
    }
};
```

##### 二维数组中的查找(剑指offer 4)
```javascript
var findNumberIn2DArray = function(matrix, target) {
    /*
        从右上角开始遍历数组，
        若target大于当前元素，下移一行
        若target小于当前元素，左移一列
        若等于直接返回
    */

    if (matrix.length == 0) {
        return false;
    }

    let row = 0;
    let column = matrix[0].length - 1;
    while (row < matrix.length && column >= 0) {
        if (target == matrix[row][column]) {
            return true;
        } else if (target > matrix[row][column]) {
            row += 1;
        } else {
            column -= 1;
        }
    }
    return false;
};

function rowBinarySearch(array, target) {
    /*
        返回刚好小于target的最大元素的索引
        若target大于所有元素，返回array长度
        若target小于所有元素，返回-1
    */
    let left = 0;
    let right = array.length - 1;
    let mid;
    let pos = 0;
    while (left <= right) {
        mid = parseInt((left + right) / 2);
        if (target < array[mid]) {
            right = mid - 1;
        } else if (target > array[mid]) {
            pos = mid;
            left = mid + 1;
        } else {
            return mid;
        }
    }
    if (left >= array.length) {
        return array.length;
    } else if (right <= -1) {
        return -1;
    } else {
        return pos;
    }
}
```

##### 数组中的逆序对(剑指offer 51)
```javascript
function merge(nums, start, end, result, count) {
    let mid = parseInt((start + end) / 2);
    let index1 = start;
    let index2 = mid + 1;
    let rindex = start;

    while(index1 <= mid && index2 <= end) {
        if (nums[index1] <= nums[index2]) {
            /** 当两个顺序数组中，左面的数组中的当前元素小于等于右面的当前元素时，逆序对数量增加。增加的量是右边数组指向当前元素的指针的偏移量。因为是有序的所以左边的当前元素一定大于之前的右边元素 **/
            count += index2 - mid - 1;
            result[rindex++] = nums[index1++];
        }else {
            result[rindex++] = nums[index2++];
        }
    }

    while (index1 <= mid) {
      /** 如果左边元素没有超出最后一个元素的位置，则说明左边有序数组的剩余的元素均大于右边的所有元素，逆序对数量每次增加右边元素的数量 **/
        result[rindex++] = nums[index1++];
        count += (end - mid);
    }
    while (index2 <= end) {
        result[rindex++] = nums[index2++];
    }
    while (start <= end) {
        nums[start] = result[start++];
    }
    return count;
}

function mergesort(nums, start, end, result, count) {
    if (start >= end){
        return 0;
    }
    let mid = parseInt((start + end) / 2);
    let lres = mergesort(nums, start, mid, result, count);
    let rres = mergesort(nums, mid + 1, end, result, count);
    count = lres + rres;

    let res = merge(nums, start, end, result, count);
    return res;
}

var reversePairs = function(nums) {
    if (nums.length <= 1) {
        return 0;
    }
    let result = new Array(nums.length);
    let count = mergesort(nums, 0, nums.length - 1, result, 0);
    return count;
};

``` 

##### 最大子矩阵(17.24)
- 二维转换为一维
``` javascript
var getMaxMatrix = function(matrix) {
    // 二维转一维
    let M = matrix.length;
    let N = matrix[0].length;
    let res = new Array(4).fill(0);
    let tempArray = new Array(N);
    let maxValue = matrix[0][0];

    for (let i = 0; i < M; i++) {
        tempArray.fill(0);
        for (let j = i; j < M; j++) {
            for (let l = 0; l < N; l++) {
                // 计算i-j行的累加和 
                tempArray[l] += matrix[j][l];
            }
            // 在tempArray中找最大子序
            let tempSum = tempArray[0];
            let tempStart = [i, 0];
            //let tempEnd = [j, 0];
            for (let k = 1; k < N; k++) {
                if (tempSum >= 0) {
                    tempSum += tempArray[k];
                } else {
                    // tempSum < 0
                    tempSum = tempArray[k];
                    // 更新的开头横坐标用应该是i，因为是累加i-j行的数值，随意开头的数据在第i行
                    tempStart = [i, k];
                    //tempEnd = [j, k];
                }
                if (tempSum > maxValue) {
                    res[0] = tempStart[0];
                    res[1] = tempStart[1];
                    res[2] = j;
                    res[3] = k;
                    maxValue = tempSum;
                }
            }
        }
    }
    return res;
};
```