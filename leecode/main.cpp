//
//  main.cpp
//  leecode
//
//  Created by Erik on 2020/4/22.
//  Copyright © 2020 Erik. All rights reserved.
//
#include "lee.hpp"
int main(int argc, const char * argv[]) {
    Solution sol = Solution();
    NNode* node5 = new NNode(5);
    NNode* node6 = new NNode(6);
    vector<NNode*> child_3{node5,node6};
    NNode* node3 = new NNode(3);
    node3->children = child_3;
    NNode* node2 = new NNode(2);
    NNode* node4 = new NNode(4);
    vector<NNode*> child_1{node3,node2,node4};
    NNode* node1 = new NNode(1);
    node1->children = child_1;
    
    sol.postorder(node1);

}
