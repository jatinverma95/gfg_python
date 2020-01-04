
def swap(arr, i, j):
    k = arr[i]
    arr[i] = arr[j]
    arr[j] = k


def reverse_string(str):
    arr = str.split('.')
    arr.reverse()

    return '.'.join(arr)


def permutation(str):
    if len(str) == 1:
        return [str]
    res = []

    for i in range(len(str)):
        first = str[i]
        second = str[:i] + str[i + 1:]

        strs = permutation(second)

        for s in strs:
            local_res = first + s
            res.append(local_res)

    return res


def main():
    import copy
    test_cases = input()
    arrs = []
    ans = []
    for t in range(int(test_cases)):
        str = input()

        ans.append(permutation(str))

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
