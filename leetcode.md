#### leetcode题解
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
