from typing import List

class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        length = len(nums)
        peak_point = False
        if len(nums) <= 2: return nums.reverse()
        #print(length)
        for i in range(length-2, -1, -1):
            if nums[i] < nums[i+1]:
                peak_point = True
                break
        if not peak_point:
            return nums.reverse()
        #print(i)

        min = -1
        for j in range(length-1, i, -1):
            #print(j, min, nums[j], nums[min], nums[i])
            if nums[j] > nums[i]:
                if min != -1:
                    if nums[min] > nums[j]:
                        min = j
                else:
                    min = j
        #print(min)

        nums[min], nums[i] = nums[i], nums[min]
        nums[i+1:] = nums[i+1:][::-1]
        return nums

s = Solution()
nums = [1,3,2]
for i in range(10):
    s.nextPermutation(nums)
    print(nums)