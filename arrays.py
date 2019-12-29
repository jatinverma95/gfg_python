def largest_sub_array_sum(arr, sum):
    print(arr, sum)
    local_sum = 0
    from_idx = -1
    to_idx = -1

    for idx, a in enumerate(arr):
        print(local_sum, a, idx)
        local_sum += a

        if from_idx == -1:
            from_idx = idx

        if local_sum > sum:
            local_sum -= arr[from_idx]
            from_idx += 1

        if local_sum == sum:
            to_idx = idx
            print(from_idx, to_idx)
            break

    print('from_idx:', from_idx, 'to_idx:', to_idx)


def counting_triplets(arr):
    arr = sorted(arr)
    ans = -1
    arr_len = len(arr)

    for a, idx in enumerate(arr):
        idx = idx - 1
        if idx < arr_len - 2:
            num = arr[arr_len - idx - 1]
            print('num', num)
            idx1 = 0
            idx2 = arr_len - idx - 2
            found = False

            while not found and idx1 != idx2:
                local_sum = arr[idx1] + arr[idx2]
                if local_sum == num:
                    print(num, arr[idx1], arr[idx2])
                    if ans == -1:
                        ans = 0
                    ans += 1
                    found = True
                elif local_sum < num:
                    idx1 += 1
                else:
                    idx2 -= 1

    print(ans)


def kadane(arr):
    import math
    output = []

    max_sum = - math.inf
    local_sum = 0
    arr_start = -1
    arr_start_local = -1
    arr_end = -1

    for idx, a in enumerate(arr):
        if arr_start == -1:
            arr_start = idx
        if arr_start_local == -1:
            if a >= 0:
                arr_start_local = idx
                local_sum = 0
        local_sum += a

        if local_sum > max_sum:
            max_sum = local_sum

            arr_start = arr_start_local
            arr_end = idx
        elif local_sum < 0:
            arr_start_local = -1

    return max_sum


def swap(arr, i, j):
    k = arr[i]
    arr[i] = arr[j]
    arr[j] = k


def missing_num(arr):
    idx = 0
    last_idx = 0

    print(len(arr), arr[idx])

    if len(arr) == 1 and arr[idx] == 1:
        return 2

    while idx < len(arr):
        print(idx)
        while arr[idx] != idx + 1:
            print(arr[idx], idx)
            if arr[idx] - 1 < len(arr):
                swap(arr, idx, arr[idx] - 1)
                last_idx = arr[idx] - 1
            else:
                return idx + 1
        idx = last_idx + 1
        last_idx = idx


def merge_2_arr_wo_space(arr1, arr2):
    import math

    total_length = len(arr1) + len(arr2)

    gap = math.ceil(total_length / 2)
    one_done = False

    while not one_done:
        if gap == 1 and not one_done:
            one_done = True

        i = 0

        while i < len(arr1):
            if i + gap < len(arr1):
                if arr1[i] > arr1[i + gap]:
                    temp = arr1[i]
                    arr1[i] = arr1[i + gap]
                    arr1[i + gap] = temp
            else:
                arr2_idx = (i + gap) - len(arr1)

                if arr2_idx < len(arr2) and arr1[i] > arr2[arr2_idx]:
                    temp = arr1[i]
                    arr1[i] = arr2[arr2_idx]
                    arr2[arr2_idx] = temp

            i += 1

        gap = math.ceil(gap / 2)

    return arr1, arr2


#   todo: Good Question
def alternate_max_min(arr):
    arr_len = len(arr)
    max_idx = arr_len - 1
    min_idx = 0
    max_elem = arr[max_idx] + 1
    ans = []

    for i in range(arr_len):

        if i % 2 == 0:
            # arr[i] = input elem(reqd aage) + num to be in output arr(x7 taaki aage remainder me na aaye aage ke liye)
            arr[i] = arr[i] + (arr[max_idx] % max_elem) * max_elem
            max_idx -= 1
        else:
            arr[i] = arr[i] + (arr[min_idx] % max_elem) * max_elem
            min_idx += 1

    for a in arr:
        ans.append(int(a / max_elem))

    return ans


def _binary_search_just_greater(arr, val, lo, hi, count_arr):
    if lo >= hi:
        if arr[lo] > val:
            return val, lo
        return None, -1

    mid = int((lo + hi) / 2)
    max_val = -1
    min_val = -1

    mid_val = arr[mid]
    if not mid - 1 < 0:
        min_val = arr[mid - 1]

    if not mid + 1 > len(arr):
        max_val = arr[mid + 1]

    if max_val > 0 or min_val > 0:
        if min_val <= val < mid_val:
            return mid_val, mid
        elif mid_val <= val < max_val:
            return max_val, mid + 1
        elif mid_val <= val >= max_val:
            return _binary_search_just_greater(arr, val, mid + 1, hi, count_arr)
        else:
            return _binary_search_just_greater(arr, val, lo, mid - 1, count_arr)
    else:
        if min_val < 0:
            if arr[mid_val] > val:
                return 2
            else:
                return 1


def power_compare(arr1, arr2):
    arr2 = sorted(arr2)
    count = 0
    count_arr = [0] * 5

    for ar in arr2:
        if ar < 5:
            count_arr[ar] += 1

    for a in arr1:
        elem = None
        idx = 0
        local_count = 0
        if a == 0:
            local_count = 0
        elif a == 1:
            if count_arr[0] > 1:
                local_count = count_arr[0]
        else:
            if a == 2:
                if count_arr[3] > 1:
                    local_count -= count_arr[3]
                if count_arr[4] > 1:
                    local_count -= count_arr[4]

            if a == 3:
                if count_arr[2] > 1:
                    local_count += count_arr[2]

            local_count += count_arr[0]
            local_count += count_arr[1]

            elem, idx = _binary_search_just_greater(arr2, a, 0, len(arr2) - 1, count_arr)

        if elem:
            count += len(arr2) - idx

        count += local_count

    return count


def merge(arr1, arr2):
    arr = []
    count = 0
    i = 0
    j = 0

    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            arr.append(arr1[i])
            i += 1
        else:
            arr.append(arr2[j])
            count += len(arr1) - i
            j += 1

    while i < len(arr1):
        arr.append(arr1[i])
        i += 1

    while j < len(arr2):
        arr.append(arr2[j])
        j += 1

    return arr, count


def count_inversion(arr, lo, hi):
    # import pdb;pdb.set_trace()
    if lo >= hi:
        if lo == hi:
            return [arr[lo]], 0
    mid = int((lo + hi) / 2)

    arr1, count1 = count_inversion(arr, lo, mid)
    arr2, count2 = count_inversion(arr, mid + 1, hi)

    arr, merge_count = merge(arr1, arr2)

    count = merge_count + count1 + count2

    return arr, count


def count_arr(arr):
    zero = 0
    one = 0
    two = 0

    for a in arr:
        if a == 0:
            zero += 1
        if a == 1:
            one += 1
        if a == 2:
            two += 1

    for i in range(one + two + zero):
        if zero > 0:
            arr[i] = 0
            zero -= 1
        elif one > 0:
            arr[i] = 1
            one -= 1
        elif two > 0:
            arr[i] = 2
            two -= 1

    return arr


def equilibrium(arr):
    i = 0
    j = len(arr) - 1
    left = arr[i]
    right = arr[j]

    while i < j:
        if left < right:
            i += 1
            left += arr[i]
        elif left > right:
            j -= 1
            right += arr[j]
        else:
            j -= 1
            i += 1
            left += arr[i]
            right += arr[j]

    if i == j and left == right:
        return i + 1

    return -1


def leader_arr(arr):
    import pdb
    pdb.set_trace()
    leaders = []
    i = len(arr) - 2
    leader_till_now = arr[len(arr) - 1]
    leaders.append(leader_till_now)

    while i >= 0:
        if arr[i] >= leader_till_now:
            leaders.append(arr[i])
            leader_till_now = arr[i]

        i -= 1

    return leaders


def minimum_platforms(arr, dep):
    dep = sorted(dep)
    i = 1
    j = 0

    min_platforms = 1
    max_platforms = 0

    while i < len(arr):
        if arr[i] <= dep[j]:
            min_platforms += 1
            i += 1
        else:
            min_platforms -= 1
            j += 1

        if min_platforms > max_platforms:
            max_platforms = min_platforms

    return max_platforms


def inverseK(arr, k):
    i = 0

    while i < len(arr):
        j = i
        m = i

        while j < len(arr) - 1 and j - m < k - 1:
            j += 1
        i = j + 1

        while m < len(arr) and m < j:
            temp = arr[m]
            arr[m] = arr[j]
            arr[j] = temp
            m += 1
            j -= 1

    return arr


def random_pivot(i, j):
    import random

    return random.randrange(i, j)


def partition(arr, lo, hi):
    i = lo - 1
    if lo < hi:
        idx = random_pivot(lo, hi)
        swap(arr, idx, hi)
    pivot = arr[hi]

    for j in range(lo, hi + 1):
        if arr[j] < pivot:
            i += 1
            swap(arr, i, j)
    swap(arr, i + 1, hi)

    return i + 1


#   todo: good ques: QuickSelect
def kSmallest_quick_select(arr, lo, hi, k):
    if lo > hi:
        return

    p = partition(arr, lo, hi)

    if p == k - 1:
        return arr[p]
    elif k - 1 < p:
        return kSmallest_quick_select(arr, lo, p - 1, k)
    else:
        return kSmallest_quick_select(arr, p + 1, hi, k)


def trapping_water(arr):
    max_left = arr[0]
    max_right = -1
    trapped_water = 0
    max_right_idx = None

    for idx, elem in enumerate(arr):
        if 0 < idx < len(arr) - 1:
            print(elem)
            if max_left <= elem:
                max_left = elem
            else:
                j = len(arr) - 1
                if (not max_right_idx) or max_right_idx <= idx:
                    max_right = -1
                    while j > idx:
                        if arr[j] > max_right:
                            max_right = arr[j]
                            max_right_idx = j
                        j -= 1

                p = min(max_right, max_left)
                if elem < p:
                    trapped_water += p - elem

                print('TW: ', trapped_water)

    return trapped_water


def trapping_water_more_optimized(arr):
    right = [0]*len(arr)
    left = [0]*len(arr)
    trapped_water = 0
    i = 1
    j = len(arr) - 2
    left[0] = arr[0]
    right[len(arr) - 1] = arr[len(arr) - 1]

    while i < len(arr):
        left[i] = max(arr[i], left[i - 1])
        i += 1

    while j >= 0:
        right[j] = max(arr[j], right[j + 1])
        j -= 1

    for idx, a in enumerate(arr):
        if a <= min(right[idx], left[idx]):
            trapped_water += min(right[idx], left[idx]) - a

    return trapped_water


def pythogoras_triplet(arr):
    arr = sorted(arr)

    for idx, a in enumerate(arr):
        arr[idx] = a * a

    i = len(arr) - 1

    while i > 0:
        lo = 0
        hi = i - 1

        while lo < hi:
            if arr[lo] + arr[hi] < arr[i]:
                lo += 1
            elif arr[lo] + arr[hi] > arr[i]:
                hi -= 1
            else:
                return True

        i -= 1

    return False


def chocolate_distribution(arr, n):
    arr = sorted(arr)
    min_diff = float("inf")
    i = len(arr) - 1

    while i >= 0 and i - n + 1 >= 0:
        diff = arr[i] - arr[i - n + 1]

        if diff < min_diff:
            min_diff = diff
        i -= 1

    return min_diff


#   todo: easy implementation for this would be take min elem from left and
#    find diff with current_elem and find max_diff
def max_diff_arr(arr):
    max_diff = -float('inf')
    current_sum = 0

    for idx in range(1, len(arr)):
        local_diff = arr[idx] - arr[idx - 1]

        if current_sum < 0:
            current_sum = local_diff
        else:
            current_sum += local_diff

        if current_sum > max_diff:
            max_diff = current_sum

    return max_diff


#   todo: good_ques
def stock_buy_sell(arr):
    local_maxima = -1
    local_minima = -1
    buy_sell = []

    i = 1

    while i < len(arr) - 1:
        while i < len(arr) and arr[i] >= arr[i + 1]:
            i += 1

        if i == len(arr):
            return False, None

        buy_sell.append({
            'buy': i,
        })
        i += 1

        while i < len(arr) and arr[i] >= arr[i - 1]:
            i += 1


def partition_first_pivot(arr, lo, hi):
    if lo == hi:
        return lo

    pivot = arr[lo]
    i = hi + 1
    j = hi

    while j >= 0:
        if arr[j] > pivot:
            i -= 1
            swap(arr, i, j)
        j -= 1

    swap(arr, i-1, lo)

    return i - 1


def quicksort_first_pivot(orig_arr, arr, lo, hi):
    if lo > hi:
        return None

    p = partition_first_pivot(arr, lo, hi)

    if arr[p] == orig_arr[p] and p != len(arr) - 1 and p != 0:
        print(arr[p], p)
        return arr[p]

    one = quicksort_first_pivot(orig_arr, arr, lo, p - 1)
    two = quicksort_first_pivot(orig_arr, arr, p + 1, hi)

    return one or two


def elem_left_small_right_greater(arr):
    ans = 0
    max_till_now = 0
    i = 1

    while i < len(arr):
        if arr[i] < arr[ans]:
            ans = -1
        if arr[i] >= arr[max_till_now]:
            max_till_now = i
            if (ans == -1 or ans == 0 or ans == len(arr) - 1) and i != len(arr) - 1:
                ans = max_till_now

        i += 1

    return arr[ans] if ans != -1 else -1


#   todo: good ques
def zig_zig(arr):
    sign = True

    for idx, a in enumerate(arr):
        if idx + 1 < len(arr):
            if sign:
                if arr[idx] > arr[idx + 1]:
                    swap(arr, idx, idx + 1)
                sign = not sign
            else:
                if arr[idx] < arr[idx + 1]:
                    swap(arr, idx, idx + 1)
                sign = not sign

    return arr


def compare_larger_no(num1, num2):
    temp = int(str(num1) + str(num2))
    temp2 = int(str(num2) + str(num1))

    if temp > temp2:
        return True
    return False


def partition_largest_no(arr, lo, hi):
    pivot = hi
    i = lo - 1
    j = lo

    while j < hi:
        if not compare_larger_no(arr[j], arr[pivot]):
            i += 1
            swap(arr, i, j)
        j += 1

    swap(arr, i + 1, pivot)

    return i + 1


def quick_sort_largest_no(arr, lo, hi):
    if lo > hi:
        return
    p = partition_largest_no(arr, lo, hi)

    quick_sort_largest_no(arr, lo, p - 1)
    quick_sort_largest_no(arr, p + 1, hi)


#   todo: good ques
def largest_formed_number(arr):
    quick_sort_largest_no(arr, 0, len(arr) - 1)
    res = ''

    print(arr)
    for a in range(len(arr) - 1, -1, -1):
        res += str(arr[a])

    return res


def spiral_print_matrix(arr):
    res = []
    is_max = True
    j = 0
    i = 0
    iidx = len(arr)
    jidx = len(arr[0])
    jidxmin = -1
    iidxmin = 0

    while not (i == iidx and j == jidx) or (i == iidxmin and j == jidxmin):
        print("####", iidx, jidx, i, j, iidxmin, jidxmin, is_max)
        print("####", res)
        import pdb
        pdb.set_trace()
        if is_max:
            print("1")
            while j < jidx:
                res.append(arr[i][j])
                j += 1
            print(res)
            j -= 1
            i += 1
            print("2")
            if i < iidx:
                while i < iidx:
                    res.append(arr[i][j])
                    i += 1

                i -= 1

            print(res)
            is_max = not is_max
            jidxmin = jidxmin + 1
            iidxmin = iidxmin + 1
            j -= 1
        else:
            print("3")
            if j >= jidxmin:
                while j >= jidxmin:
                    res.append(arr[i][j])
                    j -= 1

                j += 1
            print(res)

            print("4")
            print(iidx, jidx, i, j, iidxmin, jidxmin)
            if i >= iidxmin:
                i -= 1
                while i >= iidxmin:
                    res.append(arr[i][j])
                    i -= 1
                j += 1
                i += 1
            print(res)

            is_max = not is_max
            jidx = jidx - 1
            iidx = iidx - 1

    return res


def main():
    import copy
    test_cases = input()
    arrs = []
    ans = []
    for t in range(int(test_cases)):
        arr_len = input()
        arr_len_num = list(map(int, arr_len.split()))
        arr = input()
        a = 0
        num = list(map(int, arr.split()))

        while len(num) != 0:
            temp = []

            for b in range(arr_len_num[1]):
                temp.append(num.pop(0))
            a += 1
            arrs.append(temp)

        print(arrs)

        ans.append(spiral_print_matrix(arrs))

    for a in ans:
        print(a)
    #
    # print(ans)
    # for a1 in ans:
    #     i = len(a1) - 1
    #     while i >= 0:
    #         print(a1[i], end=" ")
    #         i -= 1
    #     print()

    # test_cases = input()
    # arrs = []
    # ans = []
    # for t in range(int(test_cases)):
    #     arr_lens = input()
    #     arr1 = input()
    #     arr2 = input()
    #     num = arr1.split()
    #     num2 = arr2.split()
    #     num = map(int, num)
    #     num2 = map(int, num2)
    #
    #     ans.append(minimum_platforms(list(num), list(num2)))
    #     # ans.append(count_inversion(num))
    #
    #     for a in ans:
    #         print(a)

    #
    # i = 0
    # while i < len(arrs):
    #     arr1, arr2 = merge_2_arr_wo_space(arrs[i], arrs[i + 1])
    #     final = arr1 + arr2
    #     ans.append(final)
    #
    #     i += 2
    #
    # for a1 in ans:
    #     for a in a1:
    #         print(a, end=" ")
    #     print()

    # arr1, arr2 = merge_2_arr_wo_space([1, 3, 5, 7], [0, 2, 6, 8, 9])
    #
    # for a1 in arr1:
    #     print(a1, end=" ")
    # print()
    # for a2 in arr2:
    #     print(a2, end=" ")


if __name__ == '__main__':
    main()
