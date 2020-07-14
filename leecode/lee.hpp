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
#include <unordered_set>
#include <queue>
#include <iostream>
#include <stack>
#include <numeric>
using namespace std;
/* 二叉树 */
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
/* N叉树 */
class NNode {
public:
    int val;
    vector<NNode*> children;

    NNode() {}

    NNode(int _val) {
        val = _val;
    }

    NNode(int _val, vector<NNode*> _children) {
        val = _val;
        children = _children;
    }
};

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
    // 面试题42. 连续子数组的最大和  动态规划  或者是分治线段树，比较复杂
    int maxSubArray(vector<int>& nums);
    // 1431. 拥有最多糖果的孩子  枚举
    vector<bool> kidsWithCandies(vector<int>& candies, int extraCandies);
    // 面试题64. 求1+2+…+n  ||的灵活运用避免使用条件语句
    int sumNums(int n);
    // 238. 除自身以外数组的乘积   前缀乘积+后缀乘积
    vector<int> productExceptSelf(vector<int>& nums);
    // 面试题29. 顺时针打印矩阵  模拟四个角
    vector<int> spiralOrder(vector<vector<int>>& matrix);
    // 128. 最长连续序列  哈希表
    int longestConsecutive(vector<int>& nums);
    // 126. 单词接龙 II  图，广度优先搜索
    vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList);
    bool one_diff(string& a, string& b);
    // 990. 等式方程的可满足性。并查集
    bool equationsPossible(vector<string>& equations);
    // 面试题46. 把数字翻译成字符串  动态规划
    int translateNum(int num);
    // 9. 回文数  不用转换字符串，/ %操作使用数字比较
    bool isPalindrome(int x);
    // 392. 判断子序列  动态规划
    bool isSubsequence(string s, string t);
    // 739. 每日温度  单调栈
    vector<int> dailyTemperatures(vector<int>& T);
    // 70. 爬楼梯 动态规划  矩阵快速幂
    int climbStairs(int n);
    // 746. 使用最小花费爬楼梯  动态规划
    int minCostClimbingStairs(vector<int>& cost);
    // 面试题 08.01. 三步问题  动态规划
    int waysToStep(int n);
    // 15. 三数之和     排序+双指针法
    vector<vector<int>> threeSum(vector<int>& nums);
    vector<int> quicksort(vector<int>& nums);
    vector<int> quicksort_join(const vector<int>& left, int mid, const vector<int>& right);
    // 面试题 17.16. 按摩师  动态规划
    int massage(vector<int>& nums);
    // 121. 买卖股票的最佳时机  动态规划
    int maxProfit(vector<int>& prices);
    // 338. 比特位计数  动态规划+位运算
    vector<int> countBits(int num);
    // 1300. 转变数组后最接近目标值的数组和
    int findBestValue(vector<int>& arr, int target);
    int sumarr(vector<int> &arr, int mini);
    // 14. 最长公共前缀
    string longestCommonPrefix(vector<string>& strs);
    // 面试题 04.02. 最小高度树
    TreeNode* sortedArrayToBST(vector<int>& nums);
    TreeNode* sortedArrayToBST_dfs(int l, int r, vector<int> nums);
    // 剑指 Offer 55 - I. 二叉树的深度
    int maxDepth(TreeNode* root);
    // 剑指 Offer 27. 二叉树的镜像
    TreeNode* mirrorTree(TreeNode* root);
    // 938. 二叉搜索树的范围和
    int rangeSumBST(TreeNode* root, int L, int R);
    // 617. 合并二叉树
    TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2);
    // 700. 二叉搜索树中的搜索
    TreeNode* searchBST(TreeNode* root, int val);
    // 590. N叉树的后序遍历
    vector<int> postorder(NNode* root);
    // 589. N叉树的前序遍历
    vector<int> preorder(NNode* root);
    void preorder_visit(NNode* root, stack<NNode*>& s, vector<int>& res);
    // 剑指 Offer 54. 二叉搜索树的第k大节点
    int kthLargest(TreeNode* root, int k);
    void kthLargest_visit(TreeNode* root, stack<TreeNode*>& s);
    // 559. N叉树的最大深度
    int maxDepthN(NNode* root);
    // 897. 递增顺序查找树
    TreeNode* increasingBST(TreeNode* root);
    void increasingBST_visit(stack<TreeNode*>& s, TreeNode* root);
};
// 297. 二叉树的序列化与反序列化

class Codec {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root);
    
    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data);
    TreeNode* rdeserialize(vector<string> list);
    TreeNode* preorder_recursion(TreeNode* root);
};

// 990. 等式方程的可满足性 并查集
class UnionFind {
private:
    vector<int> parent;
public:
    UnionFind() {
        parent.resize(26);
        iota(parent.begin(), parent.end(), 0);          //初始化，用0-25进行填充
    }
    int find(int index) {                               //寻找index对应的字母它的根节点是谁
        if (index == parent[index]) {                   //如果就是根节点，就返回根节点的索引
            return index;
        }
        parent[index] = find(parent[index]);            //如果不是，则继续向上寻找
        return parent[index];
    }
    void unite(int index1, int index2) {                //因为两者相等，所以需要联合
        parent[find(index1)] = find(index2);            //将index1的根节点指向index2的根节点
    }
};

//  146. LRU缓存机制双向链表加哈希桶
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


// 303. 区域和检索 - 数组不可变  前缀和（也算动态规划）
class NumArray {
public:
    NumArray(vector<int>& nums);
    vector<int> li;   // 不加&可以赋值成功
    int sumRange(int i, int j);
};

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray* obj = new NumArray(nums);
 * int param_1 = obj->sumRange(i,j);
 */
