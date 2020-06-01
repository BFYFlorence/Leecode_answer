//
//  main.cpp
//  leecode
//
//  Created by Erik on 2020/4/22.
//  Copyright © 2020 Erik. All rights reserved.
//
#include <iostream>
#include "lee.hpp"
int main(int argc, const char * argv[]) {
    Solution sol = Solution();
    vector<int> nums{-2,1,-3,4,-1,2,1,-5,4};
    int result = sol.maxSubArray(nums);
    cout << result << endl;
}
