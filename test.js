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

const h1 = new heap([4, 6, 8, 5, 9, 10, 22]);
console.log(h1.heap);
h1.add(40);
console.log(h1.heap);
h1.pop();
console.log(h1.heap);
console.log(h1.heap.length);