#### leetcode题解
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

##### 二叉树的前序遍历(144)
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