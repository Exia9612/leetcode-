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

console.log(findEle([1,2,3,4,5,6,7,8], 0, 7, 8))
