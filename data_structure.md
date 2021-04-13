##### Heap(堆)
###### 说明
1. 在生活中，我们处理任务的时候总是会给任务区分优先级。首先我们会处理优先级高的任务，接着我们会处理下一个优先级高的任务。这其实就是一个 优先队列 。
在很多书中会认为 堆 就是 优先队列 。但是， 优先队列 是一种抽象的数据类型，而 堆 是一种数据结构。所以 堆 并不是 优先队列 ， 堆是实现优先队列 的一种方式。
实现 优先队列 的方式有很多种，比如数组和链表。但是，这些实现方式只能保证插入操作和删除操作中的一种操作可以在O(1) 的时间复杂度内完成，而另外一个操作则需要在O(N) 的时间复杂度内完成。而 堆 能够使 优先队列 的插入操作在 O(logN) 的时间复杂度内完成，删除操作在O(logN) 的时间复杂度内完成。

2. 满足以下条件的二叉树，可以称之为 堆：
   1. 完全二叉树；
   2. 每一个节点的值都必须 大于等于或者小于等于 其孩子节点的值。堆 具有以下的特点：
      - 可以在 O(logN) 的时间复杂度内向 堆 中插入元素；
      - 可以在 O(logN) 的时间复杂度内向 堆 中删除元素；
      - 可以在 O(1) 的时间复杂度内获取 堆 中的最大值或最小值。
```javascript
class heap {
  constructor(array) {
    this.size = array.length;
    this.heapsize = array.length - 1;
    this.heap = new Array(...array);
    this._buildHeap();
  }

  _buildHeap () {
    for (let i = Math.ceil(this.heapsize / 2) - 1; i >= 0; i--) {
      this.heapify(i);
    }
  }

  heapify (i) {
    //从上到下
    let leftChild = 2 * i + 1;
    let rightChild = leftChild + 1;
    let largeChild;

    if (rightChild >= this.size || this.heap[rightChild] < this.heap[leftChild]) {
      largeChild = leftChild
    } else {
      largeChild = rightChild;
    }
    if (this.heap[i] < this.heap[largeChild]) {
      this.exchangeVal(i, largeChild);
      this.heapify(largeChild);
    }
  }

  exchangeVal(i, j) {
    let temp = this.heap[i];
    this.heap[i] = this.heap[j];
    this.heap[j] = temp;
  }

  _topUp(pos) {
    //从底向上
    if (pos == 0) {
      return;
    }
    let parentPos = Math.ceil(pos / 2 - 1);
    if (this.heap[parentPos] < this.heap[pos]) {
      this.exchangeVal(parentPos, pos);
      this._topUp(parentPos);
    }
  }

  add(ele) {
    this.heap.push(ele);
    this.heapsize++;
    this.size = this.heap.length;
    this._topUp(this.heapsize);
  }

  pop() {
    if (this.size == 0) {
      return;
    }
    //弹出最大元素
    const res = this.heap[0];
    this.heap[0] = this.heap.pop();
    this.heapsize--;
    this.size = this.heap.length;
    this.heapify(0);

    return res;
  }
}
```

