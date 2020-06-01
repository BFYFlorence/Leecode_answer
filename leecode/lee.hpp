//
//  lee.hpp
//  leecode
//
//  Created by Erik on 2020/4/22.
//  Copyright © 2020 Erik. All rights reserved.
//

#ifndef lee_hpp
#define lee_hpp

#include <stdio.h>
#include <vector>
#include <unordered_map>
#include <queue>
#include <iostream>
#include <stack>
using namespace std;


class Solution {
public:
    // 1. 两数之和  哈希
    vector<int> twoSum(vector<int>& nums, int target);
    vector<int> twoSum_improve(vector<int>& nums, int target);
    // 面试题51. 数组中的逆序对  归并排序
    int reversePairs(vector<int>& nums);
    int mergesort(vector<int>& nums, vector<int>& temp,int l, int r);
    // 23. 合并K个排序链表  优先队列
    struct ListNode {
         int val;
         ListNode *next;
         ListNode(int x) : val(x), next(NULL) {}
    };
    ListNode* mergeKLists(vector<ListNode*>& lists);
    void printlist(ListNode*);
    /*                   动态规划例子  开始
    int volume[] = {0,2,3,5,5};
    int value[] = {0,2,4,3,7};
    int N = 5;
    int V = 10;
    int array[5][11];
    for(int i=0;i<5;i++){
        for(int j=0;j<11;j++){
            array[i][j] = 0;
        }
    }
    for(int i=1;i<N;i++){
        for(int vol=volume[i];vol<=V;vol++){
            int no_pick = array[i-1][vol];
            int yes_pick = array[i-1][vol-volume[i]]+value[i];
            if(no_pick>=yes_pick){array[i][vol]=no_pick;}
            if(no_pick<=yes_pick){array[i][vol]=yes_pick;}
        }
    }
    for(int i=0;i<5;i++){
        for(int j=0;j<11;j++){
            cout << array[i][j] << endl;
        }
    }
                       动态规划例子  结束            */
    struct Status {
            int val;
            ListNode *ptr;
            bool operator < (const Status &rhs) const
        {
                return val > rhs.val;
            
        }
        };
    // 33. 搜索旋转排序数组--二分
    int search(vector<int>& nums, int target);
    int binary_search(vector<int>& nums, int target, int left, int right);
    int special_search(vector<int> &nums, int target, int left, int right);
    // 面试题56 - I. 数组中数字出现的次数  位运算
    vector<int> singleNumbers(vector<int>& nums);
    vector<int> singleNumbers_improve(vector<int>& nums);
    // 1095. 山脉数组中查找目标值  二分查找
    int findInMountainArray(int target, vector<int> &mountainArr);
    int search_mount(vector<int> &mountainArr, int left, int right);
    int binarysearch_mount(vector<int> &mountainArr, int target, int left, int right, bool);
    // 202. 快乐数
    // 我们使用 HashSet 而不是向量、列表或数组的原因是因为我们反复检查其中是否存在某数字。检查数字是否在哈希集中需要 O(1)O(1) 的时间，而对于其他数据结构，则需要 O(n)O(n) 的时间。选择正确的数据结构是解决这些问题的关键部分。
    bool isHappy(int n);
    int get_next(int n);
    // 69. x 的平方根
    // 思路一：二分法查找；
    // 思路二：牛顿法迭代，x^2-C=0，先取一点，作切线，与x轴有交点P，只要抛物线与x的右交点与P足够近[10^(-6)]
    int mySqrt(int x);
    int mySqrt_bin(int target, int left, int right);
    // 88. 合并两个有序数组  双指针法
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n);
    // 680. 验证回文字符串 Ⅱ  贪心算法
    bool validPalindrome(string s);
    // 1371. 每个元音包含偶数次的最长子字符串  前缀和+状态压缩
    int findTheLongestSubstring(string s);  // 官网的移位骚操作有点厉害
    int official_findTheLongestSubstring(string s);
    int times_findTheLongestSubstring(string s, char c);
    // 5. 最长回文子串  动态规划
    string longestPalindrome(string s);
    bool valid_palindrome(string s, int f, int l);
    // 105. 从前序与中序遍历序列构造二叉树  递归法
    struct TreeNode {
         int val;
         TreeNode *left;
         TreeNode *right;
         TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    };
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder);
    TreeNode* lr_buildTree(unordered_map<int, int> hash,vector<int>& preorder, vector<int>& inorder, int preleft, int preright,int inleft, int inright);
    
    // 76. 最小覆盖子串   滑动窗口+distance距离  while循环嵌套
    string minWindow(string s, string t);
    bool include_minWindow(string sub, string t);
    string minWindow_improve(string s, string t);
    bool check(unordered_map <char, int> ori, unordered_map <char, int> cnt) {
        for (const auto &p: ori) {
            if (cnt[p.first] < p.second) {
                return false;
            }
        }
        return true;
    }
    // 287. 寻找重复数  位运算
    int findDuplicate(vector<int>& nums);
    // 1423. 可获得的最大点数  前缀和
    int maxScore(vector<int>& cardPoints, int k);
    // 198. 打家劫舍  动态规划！！！！！！！！因为只涉及到数组中的两个元素，可以用变量来存储减少空间复杂度
    int rob(vector<int>& nums);
    // 1025. 除数博弈  动态规划
    bool divisorGame(int N);
    // 84. 柱状图中最大的矩形  使用单调栈机制
    int largestRectangleArea(vector<int>& heights);
    // 101. 对称二叉树   按照中序遍历和后续遍历依次来做
    bool isSymmetric(TreeNode* root);
    // 面试题42. 连续子数组的最大和
    int maxSubArray(vector<int>& nums);
};

//  146. LRU缓存机制
struct Node{
    Node* pre;
    Node* next;
    int key=-1;
    int value;
};
class LRUCache {
public:
    int Size;
    int current_num = 0;
    LRUCache(int capacity);
    int get(int key);
    void put(int key, int value);
    unordered_map<int, Node*> bucket;  // 直接将节点hash进bucket，这样可以避免在双向链表中寻找节点
    Node Head;
    Node Tail;
};
#endif /* lee_hpp */
