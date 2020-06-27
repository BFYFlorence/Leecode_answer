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
    Codec cod = Codec();
    
    TreeNode *root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->right->left = new TreeNode(4);
    root->right->right = new TreeNode(5);
    cod.preorder_recursion(root);

}
