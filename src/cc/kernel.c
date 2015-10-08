/* 
 * File:   kernel.c
 * Author: Rik Schaaf aka CC007 <coolcat007.nl>
 *
 * Created on Oktober 4, 2015, 4:38 PM
 */

void sumArrayHelper(double *nums, int length, int interval) {
    int index = 0;
    int next = interval / 2;
    do {
        if (next < length) {
            nums[index] += nums[next];
        }
        index += interval;
        next += interval;
    } while (index < length);
}

double sumArray(double *nums, int length) {
    if (length <= 0) {
        return 0;
    }
    int interval = 2;
    while (interval < length * 2) {
        sumArrayHelper(nums, length, interval);
        interval *= 2;
    }
    return nums[0];
}