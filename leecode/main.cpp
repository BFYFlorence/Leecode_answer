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
    unordered_map<string, int> res;
    vector<int> now{1,2,3,4,56};
    vector<int> tmp(now);
    for (int n:tmp ){
        cout << n << endl;
    }
}
