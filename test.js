function findEle(target, array, start, end) {
    //在array数组中找到比targrt小的最大元素的索引
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

let res = findEle(-933, [-1000, -938, -544, -229, 157], 0, 4);
console.log(res);