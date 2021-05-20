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

let res = findEle([1,3,5,7,9,11,13], -1);
console.log(res);