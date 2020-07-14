//
//  lee.cpp
//  leecode
//
//  Created by Erik on 2020/4/22.
//  Copyright © 2020 Erik. All rights reserved.
//
#include "lee.hpp"
vector<int> Solution::twoSum(vector<int> &nums, int target){
    vector<int> index;
    for(int i=0; i<nums.size(); i++){
        for(int j=i+1; j<nums.size(); j++){
            int result = nums.at(i)+nums.at(j);
            if(result == target){
                index.push_back(i);
                index.push_back(j);
                return index;
            }
        }
    }
    return index;
}

vector<int> Solution::twoSum_improve(vector<int> &nums, int target){
    vector<int> result;
    unordered_map<int, int> bucket;
    for(int i=0; i<nums.size(); i++){
        bucket[nums.at(i)]=i;
    }
    for(int i=0; i<nums.size(); i++){
        int another = bucket[target-nums.at(i)];
        if(another!=0 && another!=i){
            result.push_back(i);
            result.push_back(another);
            return result;
        }
    }
    return result;
}
int Solution::mergesort(vector<int>& nums, vector<int>& tmp,int l, int r){  // l、r是nums的绝对索引
    if(l>=r){
        return 0;
    }
    int mid = (l+r)/2;
    int inv_count = mergesort(nums, tmp, l, mid) + mergesort(nums, tmp, mid+1, r);
    int i=l;
    int j=mid+1;
    int pos=l;
    while (i<=mid && j<=r) {
        if(nums[i]<=nums[j]){
            tmp[pos]=nums[i];
            ++i;
            inv_count += (j-(mid+1));  // 第 1 个区间里的元素归并回去的时候，第 2 个区间已经归并回去的元素个数（j - mid - 1） 来计算逆序数的。
        }else{
            tmp[pos] = nums[j];
            ++j;
        }
        ++pos;
    }
    for(int k=i; k<=mid; ++k){
        tmp[pos++]=nums[k];  // pos++是先对pos位置赋值再++
        inv_count += (j-(mid+1));
    }
    for(int k=j; k<=r; ++k){
        tmp[pos++]=nums[k];
    }
    copy(tmp.begin() + l, tmp.begin() + r + 1, nums.begin() + l);  //没有用一个中间容器去承载排过序的数组，而是直接copy一份
    return inv_count;
}

int Solution::reversePairs(vector<int>& nums){
    int n = int(nums.size());
    vector<int> tmp(n);
    return mergesort(nums, tmp, 0, n - 1);
    /*
     Solution sol = Solution();
     vector<int> sample(5);
     sample.push_back(7);
     sample.push_back(5);
     sample.push_back(6);
     sample.push_back(4);
     int k = sol.reversePairs(sample);
     cout << k << endl;
     */
}

Solution::ListNode* Solution::mergeKLists(vector<ListNode *> &lists){
    ListNode head = ListNode(0);  // 迷就迷在不知道头部指针怎么搞
    ListNode* current = nullptr;
    current = &head;
    vector<ListNode*> currentNode;
    for(int i=0;i<lists.size();i++){
        currentNode.push_back(lists[i]);
    }
    long n = 0;
    while (n<lists.size()) {
        int min = 0;
        int currentindex = 0;
        for (int i=0; i<currentNode.size(); i++) {
            if (currentNode[i] != NULL) {
                min = currentNode[i]->val;
                break;
            }
        }
        for (int i=0; i<currentNode.size(); i++) {
            if (currentNode[i] != NULL && currentNode[i]->val <= min) {
                min = currentNode[i]->val;
                currentindex = i;
            }
        }
        if (currentNode[currentindex]!=NULL) {
            if (currentNode[currentindex]->next==NULL) {
                currentNode[currentindex] = NULL;
                n++;
                ListNode* newnode = new ListNode(min);
                current->next = newnode;
                current = newnode;
            }else{
                currentNode[currentindex] = currentNode[currentindex]->next;
                ListNode* newnode = new ListNode(min);
                current->next = newnode;
                current = newnode;
            }
        }else{
            n++;
        }
    }
    return head.next;
}
/*ListNode head1 = ListNode(0);
ListNode newnode1 = ListNode(1);  // 带有类名的初始化之后，变量拥有唯一的地址，所以需要声明三个独立的变量来赋值
                                  // 否则会出错，只以最后一个有效。 new 方法用来创造新的地址!
ListNode newnode2 = ListNode(3);
ListNode newnode3 = ListNode(4);
ListNode* curr1 = &head1;

curr1->next = &newnode1;
curr1 = &newnode1;

curr1->next = &newnode2;
curr1 = &newnode2;

curr1->next = &newnode3;
curr1 = &newnode3;
//sol.printlist(head1.next);

ListNode head2 = ListNode(0);
ListNode newnode4 = ListNode(1);
ListNode newnode5 = ListNode(4);
ListNode newnode6 = ListNode(5);
ListNode* curr2 = &head2;

curr2->next = &newnode4;
curr2 = &newnode4;
curr2->next = &newnode5;
curr2 = &newnode5;
curr2->next = &newnode6;
curr2 = &newnode6;
//sol.printlist(head2.next);

ListNode head3 = ListNode(0);
ListNode newnode7 = ListNode(2);
ListNode newnode8 = ListNode(6);
ListNode* curr3 = &head3;
curr3->next = &newnode7;
curr3 = &newnode7;
curr3->next = &newnode8;
curr3 = &newnode8;
//sol.printlist(head3.next);

vector<ListNode*> lists;
lists.push_back(head1.next);
lists.push_back(head2.next);
lists.push_back(head3.next);*/
void Solution::printlist(ListNode* ptr){
    while (ptr->next!=NULL) {
        //cout << "begin print" << endl;
        cout << ptr->val << endl;
        ptr = ptr->next;
    }
    cout << ptr->val << endl;
}

int Solution::search(vector<int> &nums, int target){
    if (nums.size()==0) {
        return -1;
    }
    return special_search(nums, target, 0, int(nums.size())-1);
}


int Solution::special_search(vector<int> &nums, int target, int left, int right){
    if (left==right) {
        if (nums[left]==target) {
            return left;
        }else{return -1;}
    }
    if (right-left==1) {
    if (nums[left]==target) {return left;}
    if (nums[right]==target) {return right;}}
    
    int mid = nums[(left+right)/2];
    int ini = nums[left];
    if (mid>ini) {
        if (target>=mid) {
            return special_search(nums, target, (left+right)/2, right);  // 需要再分
        }else{
            if (target<ini) {return special_search(nums, target, (left+right)/2, right);}
            if (target>=ini) {return binary_search(nums, target, left, (left+right)/2);}
        }
    }else{
        if (target>=mid && target<ini) {
            return binary_search(nums, target, (left+right)/2, right);
        }else{
            return special_search(nums, target, left, (left+right)/2);
        }
    }
return -1;
}

int Solution::binary_search(vector<int> &nums, int target, int left, int right){
    if (right-left==1) {
        if (nums[left]==target) {return left;}
        if (nums[right]==target) {return right;}
        return -1;
    }else{
        int mid = nums[(left+right)/2];
        if (mid==target){return (left+right)/2;}
        if (target>mid){return binary_search(nums, target, (left+right)/2, right);}
        if (target<mid){return binary_search(nums, target, left, (left+right)/2);}
        return -1;
    }
}

struct singlenode{
    int val;
    int times;
    singlenode(int b){
        val=b;
        times=1;
    }
    bool operator<(const singlenode& sn1)const{
        return times < sn1.times;
    }
};
vector<int> Solution::singleNumbers(vector<int>& nums){
    unordered_map<int, int> q;
    vector<int> result;
    for (int i=0; i<nums.size(); i++) {
        switch (q[nums[i]]) {
            case 0:
                q[nums[i]]=1;
                break;
            case 1:
                q.erase(nums[i]);
                break;
            default:
                break;
        }
    }
    for (auto it=q.begin(); it!=q.end(); it++) {
        result.push_back(it->first);
    }
    return result;
}

vector<int> Solution::singleNumbers_improve(vector<int> &nums){  // 位运算，有点绕，但是仔细想可以想明白
    int result = 0;
    for (int i=0; i<nums.size(); i++) {
        result ^= nums[i];
    }
    int div=1;
    while ((div&result)==0) {  // 找到结果中从低位数第一个为1的二进制数div
        div<<=1;
    }
    int a=0;
    int b=0;
    for(int n:nums){
        if (n&div) {
            a ^= n;
        }else{
            b ^= n;
        }
    }
    return vector<int>{a,b};
}

int Solution::findInMountainArray(int target, vector<int> &mountainArr){
    int midx = search_mount(mountainArr, 0, int(mountainArr.size())-1);
    int mid = mountainArr[midx];
    if (target>mid) {
        return -1;
    }
    if (target==mid) {
        return midx;
    }else{
        int left = binarysearch_mount(mountainArr, target, 0, midx, true);
        if (left!=-1) {
            return left;
        }else{
            int right = binarysearch_mount(mountainArr, target, midx+1, int(mountainArr.size()), false);
            if (right!=-1) {return right;}else{return -1;}
        }
    }
}

int Solution::search_mount(vector<int> &mountainArr, int left, int right){
    int mid = (left+right)/2;
    if (mountainArr[mid-1]<mountainArr[mid] && mountainArr[mid]>mountainArr[mid+1]) {
        return mid;
    }
    if (mountainArr[mid-1]>mountainArr[mid]) {
        return search_mount(mountainArr, left, mid);
    }
    if (mountainArr[mid]<mountainArr[mid+1]) {
        return search_mount(mountainArr, mid, right);
    }
    return -1;
}

int Solution::binarysearch_mount(vector<int> &mountainArr, int target, int left, int right, bool isincrease){
    int midx = (left+right)/2;
    if (right-left==1) {
        if (mountainArr[left]==target) {
            return left;
        }else{return -1;}
    }
    if (right==left) {
        if (mountainArr[left]==target) {
            return left;
        }else{return -1;}
    }
    if (mountainArr[midx]==target) {
        return midx;
    }
    if (isincrease) {
        if (mountainArr[midx]>target) {
            return binarysearch_mount(mountainArr, target, left, midx, true);
        }else{
            return binarysearch_mount(mountainArr, target, midx+1, right, true);
        }
    }else{
        if (mountainArr[midx]>target) {
            return binarysearch_mount(mountainArr, target, midx+1, right, false);
        }else{
            return binarysearch_mount(mountainArr, target, left, midx, false);
        }
    }
}


bool Solution::isHappy(int n){
    int slow = n;
    int fast = get_next(n);
    while (fast!=1 && slow!=fast) {
        slow = get_next(slow);
        fast = get_next(get_next(fast));
    }
    return true;
}

int Solution::get_next(int n){
    int sum = 0;
    while (n>0) {
        sum += (n%10)*(n%10);
        n /= 10;
    }
    return sum;
}

int Solution::mySqrt(int x){
    if (x==1) {
        return 1;
    }
    return mySqrt_bin(x, 0, x);
}

int Solution::mySqrt_bin(int target, int left, int right){
    long mid = (left+right)/2;
    if (mid*mid==target) {
        return int(mid);
    }else{
        if (right-left==1) {
            return left;
        }
    }
    if (mid*mid<target) {
        return mySqrt_bin(target, mid, right);
    }else{
        return mySqrt_bin(target, left, mid);
    }
}

void Solution::merge(vector<int>& nums1, int m, vector<int>& nums2, int n){
    int l = int(nums1.size())-1;
    int p=m-1;
    int q=n-1;
    while (p>=0 && q>=0) {
        if (nums1[p]>=nums2[q]) {
            nums1[l] = nums1.at(p);
            l--;
            p--;
        }else{
            nums1[l] = nums2.at(q);
            l--;
            q--;
        }
    }
    while (q>=0) {
        nums1[l] = nums2.at(q);
        l--;
        q--;
    }
    while (p>=0) {
        nums1[l] = nums1.at(p);
        l--;
        p--;
    }
}

bool Solution::validPalindrome(string s){  //分两个方向
    int f=0;
    int l=int(s.size())-1;
    bool choice = true;
    int break_f,break_l;
    bool result = true;
    while (f<l) {
        if (s[f]!=s[l]) {
            if (choice) {
                break_f = f;
                break_l = l;
                f++;
                choice=false;
            }else{
                result = false;
                break;
            }
        }else{
            f++;
            l--;
        }
    }
    cout << "first:" << result <<endl;
    if (result) {
        return result;
    }else{
        f = break_f;
        l = break_l;
        choice  = true;
        result = true;
        //int o=1;             在else内部再声明变量的话，外部用不到，就算是相同名字的变量也是这样
        while (f<l) {
            if (s[f]!=s[l]) {
                if (choice) {
                    l--;
                    choice=false;
                }else{
                    result = false;
                    break;
                }
            }else{
                f++;
                l--;
            }
        }
    }
    //cout << o << endl;
    return result;
}

int Solution::findTheLongestSubstring(string s){
    string vowel = "aeiou";
    int length = int(s.length());
    for (int i=length; length>0; i--) {
        for (int j=0; j<=length-i; j++) {
            string sub = s.substr(j,i);
            bool even = true;
            for (int k=0; k<vowel.size(); k++) {
                int times = times_findTheLongestSubstring(sub, vowel[k]);
                if (times%2==1) {
                    even = false;
                    break;
                }
            }
            if (even) {
                return i;
            }
        }
    }
    return 0;
}

int Solution::times_findTheLongestSubstring(string s, char c){
    int times = 0;
    for (int i=0; i<s.size(); i++) {
        if (s[i]==c) {
            times++;
        }
    }
    return times;
}

int Solution::official_findTheLongestSubstring(string s){
    int ans = 0, status = 0, n = int(s.length());
    vector<int> pos(1 << 5, -1);  // 长度为2^5=32的，值都为-1的一位数组
    pos[0] = 0;  // 把第一位赋值为0
    for (int i = 0; i < n; ++i) {
        if (s[i] == 'a') {
            status ^= 1<<0;                 //           u o i e a
        } else if (s[i] == 'e') {           //  status   0 0 0 0 1    1<<0
            status ^= 1<<1;                 //         ^ 0 0 0 1 0    1<<1
        } else if (s[i] == 'i') {           //  result   0 0 0 1 1
            status ^= 1<<2;
        } else if (s[i] == 'o') {
            status ^= 1<<3;
        } else if (s[i] == 'u') {
            status ^= 1<<4;
        }
        if (~pos[status]) {  // pos[status]!=-1
            ans = max(ans, i + 1 - pos[status]);  //  奇偶性相同的位置，取它们最长的长度
        //  注意，当status为0的时候，pos[status]永远为0，代表i之前的所有字符
            // 当status不为0时，则是代表奇数个子串，需要相减才能得到满足条件的子串
        } else {
            pos[status] = i + 1;  //第一次碰到status的情况，直接赋值子串长度
        }
    }
    return ans;
}

TreeNode* Solution::buildTree(vector<int> &preorder, vector<int> &inorder){
    if (preorder.size()==0) {
        return NULL;
    }
    unordered_map<int, int> hash;
    for (int i=0; i<inorder.size(); i++) {  // 用哈希桶，以空间换取时间
        hash.insert({inorder[i],i});
    }
    TreeNode* root = new TreeNode(-1);
    root->left = lr_buildTree(hash, preorder, inorder, 0, int(preorder.size()-1), 0, inorder.size()-1);
    return root->left;
}

TreeNode* Solution::lr_buildTree(unordered_map<int, int> hash, vector<int> &preorder, vector<int> &inorder, int preleft, int preright, int inleft, int inright){
    
    int root_idx = 0;  // 中序遍历中根节点的索引值
    
    TreeNode* root = new TreeNode(preorder[preleft]);  // 先序遍历中第一个值即为根节点的值
    root_idx = hash.at(root->val);
    
    // 还有一种是不需要if条件语句，而是直接遍历到最后一个元素，只能是根节点
    if (root_idx-inleft==1) {  // 如果中序遍历中，根节点左边只有一个值，则这个值直接成为根节点的左节点
        TreeNode* node = new TreeNode(inorder[root_idx-1]);
        root->left = node;
    }else{
        if (root_idx-inleft>1) {    // 如果说不止一个值，那么继续递归
            root->left = lr_buildTree(hash, preorder, inorder, preleft+1, root_idx-inleft+preleft, inleft, root_idx-1);
        }
    }
    
    if (inright-root_idx==1) {
        TreeNode* node = new TreeNode(inorder[root_idx+1]);
        root->right = node;
    }else{
        if (inright-root_idx>1) {
            root->right = lr_buildTree(hash, preorder, inorder, preright+root_idx-inright+1, preright, root_idx+1, inright);
        }
    }
    return root;
}

string Solution::minWindow(string s, string t){
    int t_length = int(t.size());
    unordered_map<char, int> T;
    for (int i=0; i<t.size(); i++) {
        if (T.find(t[i])==T.end()) {
            T.insert({t[i],1});
        }else{
            T.at(t[i])++;
        }
    }
    /*
    for (auto it=T.begin(); it!=T.end(); it++) {
        cout << it->first << ":" << it->second << endl;
    }*/
    for (int i=t_length; i<s.size()+1; i++) {
        for (int j=0; j<s.size()-i+1; j++) {
            if (T.find(s[j])==T.end()) {
                continue;
            }
            string sub = s.substr(j,i);
            cout << sub << ":" << s[j] << endl;
            if (include_minWindow(sub, t)) {
                return sub;
            }
        }
    }
    return "";
}

bool Solution::include_minWindow(string sub, string t){
    unordered_map<char, int> T;
    unordered_map<char, int>::const_iterator got;
    for (int i=0; i<t.size(); i++) {    //使用字典来存储t中字符出现的次数
        if (T.find(t[i])==T.end()) {
            T.insert({t[i],1});
        }else{
            T.at(t[i])++;
        }
    }
    for (int i=0; i<sub.size(); i++) {  //扫描sub子串，并且计数目标字符出现的次数
        char target = sub[i];
        got = T.find(target);
        if (got!=T.end()) {
            T.at(target)--;
        }
    }
    for (auto it=T.begin(); it!=T.end(); it++) {    // 如果说字符出现次数不够，说明不符合，返回false
        if (it->second>0) {
            return false;
        }
    }
    return true;
}


string Solution::minWindow_improve(string s, string t){
        unordered_map <char, int> ori, cnt;
        for (const auto &c: t) {
            ++ori[c];
        }
        int l = 0, r = -1;
    int len = INT_MAX, ansL = -1;//, ansR = -1;

        while (r < int(s.size())) {
            if (ori.find(s[++r]) != ori.end()) {
                ++cnt[s[r]];
            }
            while (Solution::check(ori, cnt) && l <= r) {  //右指针向右移动时的判断条件没有掌握好，本来想着while循环嵌套，但是没写出来。
                if (r - l + 1 < len) {
                    len = r - l + 1;
                    ansL = l;
                }
                if (ori.find(s[l]) != ori.end()) {
                    --cnt[s[l]];
                }
                ++l;
            }
        }
        return ansL == -1 ? string() : s.substr(ansL, len);
    }

LRUCache::LRUCache(int capacity){
    Head.next = &Tail;
    Tail.pre = &Head;
    Size = capacity;
}

void LRUCache::put(int key, int value){
    if (bucket.find(key)==bucket.end()) {   // 桶里找不到
        if(current_num<Size){  //数量不大于最大数量
            Node* node = new Node;
            bucket.insert({key,node});
            node->value = value;
            node->key = key;
            node->next = Head.next;
            node->pre = &Head;
            Head.next->pre = node;
            Head.next = node;
            current_num++;
        }else{  // 数量等于最大数量
            bucket.erase(Tail.pre->key);
            Tail.pre = Tail.pre->pre;
            Tail.pre->next->next = nullptr;
            Tail.pre->next->pre = nullptr;
            Tail.pre->next = &Tail;
            Node* node = new Node;
            
            bucket.insert({key,node});
            node->value = value;
            node->key = key;
            node->next = Head.next;
            node->pre = &Head;
            Head.next->pre = node;
            Head.next = node;
        }
    }else{                                  // 桶里找得到
        bucket[key]->value = value; 
        Node* search_node = bucket[key];
        search_node->next->pre = search_node->pre;
        search_node->pre->next = search_node->next;
        search_node->value = value;
        search_node->next = Head.next;
        search_node->pre = &Head;
        Head.next->pre = search_node;
        Head.next = search_node;
    }
    cout << bucket.size() << endl;
}

int LRUCache::get(int key){
    if (bucket.find(key)!=bucket.end()) {
        Node* search_node = bucket[key];
        search_node->next->pre = search_node->pre;
        search_node->pre->next = search_node->next;
        search_node->next = Head.next;
        search_node->pre = &Head;
        Head.next->pre = search_node;
        Head.next = search_node;

        return bucket[key]->value;
    }else{
        return -1;
    }
}

int Solution::findDuplicate(vector<int> &nums){  // 最大值就是nums的长度向下数
    // 不可以把nums里面的值直接位运算，这样会极大地溢出
    long n = nums.size(), ans = 0;
    // 确定二进制下最高位是多少，这点很重要
    long bit_max = 31;
    while (!((n - 1) >> bit_max)) {
        bit_max -= 1;
    }
    for (int bit = 0; bit <= bit_max; ++bit) {  //从第一位开始计算
        int x = 0, y = 0;
        for (int i = 0; i < n; ++i) {  //每一次都遍历完全
            if (nums[i] & (1 << bit)) {  // 如果说那一位是1就x++
                x += 1;
            }
            if (i >= 1 && (i & (1 << bit))) {  // 如果说i属于1-n，就依次y++
                y += 1;
            }
        }
        if (x > y) {
            ans |= 1 << bit;
        }
    }
    return ans;
}

int Solution::maxScore(vector<int> &cardPoints, int k){
    /*  滑动窗口会超时
    int result = 0;
    int length = int(cardPoints.size());
    int window_size = length - k;
    for (int i=0; i<k+1; i++) {
        int sum = 0;
        int f = 0;
        int l = length-1;
        while (f<i) {
            sum += cardPoints[f];
            f++;
        }
        while (l>i+window_size-1) {
            sum += cardPoints[l];
            l--;
        }
        cout << sum <<endl;
        if (sum>result) {
            result = sum;
        }
    }
    return result;
     */
    
    int length = int(cardPoints.size());
    vector<int> pre_sum(length,0);
    cout << pre_sum.size() << endl;
    int sum = 0;
    for (int i=0; i<length; i++) {
        sum += cardPoints[i];
        pre_sum[i] = sum;
    }
    int total = pre_sum[length-1];
    
    int result = 0;
    for (int i=length-1; i>=length-k-1; i--) {  // 以后遇见size还是手动转为int吧，不然在比较的时候会因为unsigned类型而出现差错
        //cout << "i:" << i << endl;
        int div = 0;
        int index = i+k-length;
        if (index==-1) {
            div = total-pre_sum[i];
        }else{
            div = total-(pre_sum[i] - pre_sum[i+k-length]);
            //cout << "i+k-length:" << i+k-length << endl;
        }
        if (div>result) {
            result = div;
        }
    }
    return result;
}

string Solution::longestPalindrome(string s){  //动态规划
    vector<vector<bool>> dp(s.size(),vector<bool>(s.size()));
    for (int i=0; i<s.size(); i++) {
        dp[i][i] = true;
    }
    int maxlen = 1;
    int begin = 0;
    for (int j=1; j<s.size(); j++) {
        for (int i=0; i<j; i++) {
            if (s[i]!=s[j]) {
                dp[i][j] = false;
            }else{
                if (j-i<3) {
                    dp[i][j] = true;
                }else{
                    dp[i][j] = dp[i+1][j-1];
                }
            }
            if (dp[i][j] && j-i+1>maxlen) {
                maxlen = j-i+1;
                begin = i;
            }
        }
    }
    return s.substr(begin,maxlen);
}

int Solution::rob(vector<int> &nums){
    int length = int(nums.size());
    if (length==0) {
        return NULL;
    }else{
        if (length==1) {
            return nums.front();
        }
    }
    int first;
    int second;
    int temp;
    first = nums.at(0);
    second = max(nums.at(0),nums.at(1));
    for (int i=2; i<int(nums.size()); i++) {
        temp = second;                          //因为要更新second，所以说先用临时值存储
        second = max(first+nums.at(i), second); //再进行更新
        first = temp;                           //再赋值
    }
    return max(first, second);
}

bool Solution::divisorGame(int N){
    vector<bool> dp(N+1, false);
    dp[0] = false;
    dp[1] = false;
    dp[2] = true;
    for (int i=0; i<=N; i++) {
        for (int j=1; j<i; j++) {
            if (i%j==0 && dp[i-j]==false) {
                dp[i] = true;
                break;
            }
        }
    }
    return dp[N];
}

int Solution::largestRectangleArea(vector<int> &heights){
    /*  超出时限
    int length = int(heights.size());
    int result = 0;
    for (int i=0; i<length; i++) {
        // cout << "i:" << i << endl;
        int width = 1;
        int r = i;
        int l = i;
        while (r+1<length && heights[r+1]>=heights[i]) {
            width++;
            r++;
        }
        while (l-1>=0 && heights[l-1]>=heights[i]) {
            width++;
            l--;
        }
        // cout << "width:" << width << endl;
        int S = heights[i]*width;
        if (S>result) {
            result = S;
        }
    }
    return result;*/
    int length = int(heights.size());
    stack<int> mono_stack;
    vector<int> left(length);
    vector<int> right(length);
    int result = 0;
    
    for (int i=0; i<length; i++) {  // 从左边遍历
        while (!mono_stack.empty() && heights[mono_stack.top()]>=heights[i]) {
            mono_stack.pop();
        }
        left[i] = (mono_stack.empty() ? -1:mono_stack.top());
        mono_stack.push(i);
    }
    
    mono_stack = stack<int>();
    for (int j = length - 1; j >= 0; j--) {// 从右边遍历
        while (!mono_stack.empty() && heights[mono_stack.top()]>=heights[j]) {
            mono_stack.pop();
        }
        right[j] = (mono_stack.empty() ? length:mono_stack.top());
        mono_stack.push(j);
    }
    for (int k=0; k<length; k++) {
        int S = (right[k]-left[k]-1)*heights[k];
        result = (S>result ? S:result);
    }
    return result;
}

bool Solution::isSymmetric(TreeNode *root){  //关键是中序遍历和后续遍历要理解，有递归输出和迭代输出两种
    //TreeNode* Root = root;
    TreeNode* rl = root;
    TreeNode* rr = root;
    stack<TreeNode*> tree_left;
    stack<TreeNode*> tree_right;
    while ((rr!=nullptr && rl!=nullptr) || !tree_left.empty() || !tree_right.empty()) {
        if (rr!=nullptr && rl!=nullptr) {
            tree_right.push(rr);
            tree_left.push(rl);
            rr = rr->right;
            rl = rl->left;
        }else{
            if ((rr==nullptr && rl!=nullptr) || (rl==nullptr && rr!=nullptr)) {
                return false;
            }
            rl = tree_left.top();
            rr = tree_right.top();
            if (rl->val!=rr->val) {
                return false;
            }
            
            tree_left.pop();
            tree_right.pop();
            
            
            rl = rl->right;
            rr = rr->left;
        }
        
    }

    return true;
}

int Solution::maxSubArray(vector<int> &nums){
    int length = int(nums.size());
    vector<int> res(length);  // res[i]为以第i个元素结尾的连续数组的最大和，不应该定义成前缀和
    res[0] = nums[0];
    int result = res[0];
    for (int i=1; i<length; i++) {
        // 要判断res[i]是否大于零，大于零的时候才可以考虑进去
        res[i] = (res[i-1]>0 ? res[i-1]+nums[i]:nums[i]);
        cout << res[i] << endl;
        result = (res[i]>result ? res[i]:result);
    }
    return result;
}

vector<bool> Solution::kidsWithCandies(vector<int> &candies, int extraCandies){
    int max_value = 0;
    int length = int(candies.size());
    vector<bool> res(length,false);
    for (int i=0; i<length; i++) {
        max_value = (candies[i]>max_value ? candies[i]:max_value);
    }
    for (int i=0; i<length; i++) {
        res[i] = (candies[i]+extraCandies>=max_value ? true:false);
    }
    return res;
}

int Solution::sumNums(int n){
    n && (n+=sumNums(n-1));
    return n;
}

int NumArray::sumRange(int i, int j){
    // 可以插入一个虚拟 0 作为li数组中的第一个元素。这个技巧可以避免在 sumrange 函数中进行额外的条件检查
    if (i!=0) {
        return li[j]-li[i-1];
    }else{
        if (j==0) {
            return li[0];
        }else{
            return li[j];
        }
    }
    
}

NumArray::NumArray(vector<int>& nums){
    int res = 0;
    for (int i=0; i<int(nums.size()); i++) {
        res += nums[i];
        li.push_back(res);
    }
}

vector<int> Solution::productExceptSelf(vector<int> &nums){
    int length = int(nums.size());
    vector<int> pre(length);
    pre[0] = 1;
    vector<int> suf(length+1);
    suf[length] = 1;
    for (int i=0; i<length-1; i++) {
        pre[i+1] = nums[i]*pre[i];
    }
    for (int i=length-1; i>=0; i--) {       // 在计算后缀积的时候还可以接着计算结果乘积
        suf[i] = nums[i]*suf[i+1];
    }
    vector<int> res(length);  // 可以直接拿pre数组存储结果，不用另外申请
    for (int i=0; i<length; i++) {
        res[i] = pre[i]*suf[i+1];
    }
    return res;
}

vector<int> Solution::spiralOrder(vector<vector<int> > &matrix){
    vector<int> res;
    if (matrix.size()==0) {
        return res;
    }
    int length_o = int(matrix.size());
    int length_i = int(matrix.front().size());
    int top=0;
    int bottom=length_o-1;
    int left=0;
    int right=length_i-1;
    if (top==bottom && left==right) {
        res.push_back(matrix[0][0]);
        return res;
    }else{
        if (top==bottom) {
            for (int i=0; i<right; i++) {
            res.push_back(matrix[0][i]);
        }
            return res;
    }
        if (left==right) {
            for (int i=0; i<bottom; i++) {
                res.push_back(matrix[i][0]);
            }
            return res;
        }
    }
    while (bottom-top>=0 || right-left>=0) {
        for (int i=left; i<=right; i++) {
            cout << matrix[top][i] << endl;
            res.push_back(matrix[top][i]);
        }
        for (int i=top+1; i<=bottom; i++) {
            cout << matrix[i][right] << endl;
            res.push_back(matrix[i][right]);
        }
        if (bottom-top>0) {
            for (int i=right-1; i>=left; i--) {
                cout << matrix[bottom][i] << endl;
                res.push_back(matrix[bottom][i]);
            }
        }
        if (right-left>0) {
            for (int i=bottom-1; i>=top+1; i--) {
                cout << matrix[i][left] << endl;
                res.push_back(matrix[i][left]);
            }

        }
        top++;
        bottom--;
        left++;
        right--;
    }
    return res;
}


int Solution::longestConsecutive(vector<int> &nums){
    unordered_set<int> set;
    int res=0;
    for (int n:nums) {
        set.insert(n);  // 有重复
    }
    int currnum;
    int l;
    for (int n:nums) {
        if (set.find(n-1)==set.end()) {  // 这一步优化可以避免无意义的循环
            currnum = n;
            //cout << currnum << endl;
            l = 1;
            while (set.find(currnum+1)!=set.end()) {
                l++;
                currnum++;
            }
            res = (l>res ? l:res);
        }
    }
    return res;
}

vector<vector<string>> Solution::findLadders(string beginWord, string endWord, vector<string> &wordList){
    const int INF = 1 << 20;
    unordered_map<string, int> wordId;
    vector<string> idWord;
    vector<vector<int>> edges;
    int id = 0;                                             //为每个单词创建一个ID，使用哈希表实现
    for (const string& word : wordList) {                   
        if (!wordId.count(word)) {                          //如果哈希表中找不到对应单词
            wordId[word] = id++;                            //则为此单词映射一个ID，先赋值，再++
            idWord.push_back(word);
        }
    }
    if (!wordId.count(endWord)) {
        return {};
    }
    if (!wordId.count(beginWord)) {                         //把beginWord也加入
        wordId[beginWord] = id++;
        idWord.push_back(beginWord);
    }
    
    edges.resize(idWord.size());                            //例子中size=6，edges存储所有距离为1的单词
    for (int i = 0; i < idWord.size(); i++) {
        for (int j = i + 1; j < idWord.size(); j++) {
            if (one_diff(idWord[i], idWord[j])) {
                edges[i].push_back(j);
                edges[j].push_back(i);                      //得到对应单词相连的单词，是个二维
            }
        }
    }
    const int dest = wordId[endWord];                       //保存尾词的索引，即为终点
    vector<vector<string>> res;                             //保存结果
    queue<vector<int>> q;                                   //创建队列
    
    vector<int> cost(id, INF);                              //cost[i]表示beginWord对应的点
                                                            //到第i个点的代价（即转换次数）
                                                            //初始情况下其所有元素初始化为无穷大
    q.push(vector<int>{wordId[beginWord]});
    cost[wordId[beginWord]] = 0;
    while (!q.empty()) {
        vector<int> now = q.front();
        q.pop();
        int last = now.back();                              //得到数组的最后一个单元的引用，此时为单词id
        if (last == dest) {                                 //如果到达了终点
            vector<string> tmp;
            for (int index : now) {
                tmp.push_back(idWord[index]);
            }
            res.push_back(tmp);                             //保存结果
        } else {
            for (int i = 0; i < edges[last].size(); i++) {  //遍历和最后一个单词距离为1的单词数组
                int to = edges[last][i];                    //得到这个单词在单词表中的索引，也即为id
                if (cost[last] + 1 <= cost[to]) {           //这个判断用于保存最短路径
                    cost[to] = cost[last] + 1;
                    vector<int> tmp(now);
                    tmp.push_back(to);
                    q.push(tmp);
                }
            }
        }
    }
    return res;
}

bool Solution::one_diff(string& a, string& b){
    int length = int(a.size());
    int diff=0;
    for (int i=0; i<length; i++) {
        if (a[i]!=b[i]) {
            diff++;
        }
    }
    return (diff==1 ? true:false);
}

bool Solution::equationsPossible(vector<string> &equations){
    UnionFind uf;                                              // 声明根节点
    for (const string& str: equations) {
        if (str[1] == '=') {
            int index1 = str[0] - 'a';                         //获取字符的相对位置，若为a，则index为0
            int index2 = str[3] - 'a';
            uf.unite(index1, index2);
        }
    }
    for (const string& str: equations) {
        if (str[1] == '!') {
            int index1 = str[0] - 'a';
            int index2 = str[3] - 'a';
            if (uf.find(index1) == uf.find(index2)) {
                return false;
            }
        }
    }
    return true;
}

int Solution::translateNum(int num){            //如果使用递归，返回的即是结果数量1等等
    string s = to_string(num);
    int length = int(s.size());
    int p=0, q=0, r=1;                          //r保存结果数量
    for (int i=0; i<length; i++) {
        p=q;                                    //p保存i-2前缀方法数量
        q=r;                                    //q保存i-1前缀方法数量
        if (i==0) {
            continue;
        }
        auto pre = s.substr(i-1,2);
        if (pre<="25" && pre>="10") {
            r+=p;
        }
    }
    return r;
}

bool Solution::isPalindrome(int x){
    string s = to_string(x);
    int length = int(s.size());
    int l=0, r=length-1;
    while (l<=length-2 && r>=1) {
        if (s[l]!=s[r]) {
            return false;
        }
        l++;
        r--;
    }
    return true;
}

bool Solution::isSubsequence(string s, string t){
    int length = int(s.size());                         // 短串的长度
    int L = int(t.size());                              // 长串的长度
    vector<bool> res(length+1, false);
    res[0] = true;
    int idx = 0;
    for (int i=0; i<length; i++) {
        while (idx<L) {
            if (t[idx]==s[i]) {
                res[i+1] = res[i];                      // 状态转移方程
                break;
            }else{
                idx++;
            }
        }
    }
    return res[length];
}

vector<int> Solution::dailyTemperatures(vector<int> &T){
    /*
             int n = T.size();
             vector<int> ans(n);
             stack<int> s;
             for (int i = 0; i < n; ++i) {
                 while (!s.empty() && T[i] > T[s.top()]) {
                     int previousIndex = s.top();
                     ans[previousIndex] = i - previousIndex;
                     s.pop();
                 }
                 s.push(i);
             }
             return ans;
     */
    stack<int> S;
    vector<int> res(int(T.size()));
    S.push(0);
    int idx = 1;
    while (!S.empty()) {
        while (!S.empty() && T[idx]>T[S.top()]) {
            int tmp = S.top();                              //使用tmp保存S.top()结果之后，发现运行效率增加
            res[tmp]=idx-tmp;                               //推测是因为top方法较慢，最好用中间值保存
            S.pop();
        }
        S.push(idx);
        if (idx==T.size()-1) {
            break;
        }
        idx++;
    }
    return res;
}

int Solution::climbStairs(int n){
    int first = 1;
    int second = 1;
    for (int i=2; i<=n; i++) {
        int tmp = second;
        second = tmp+first;
        first = tmp;
    }
    return second;
}

int Solution::minCostClimbingStairs(vector<int> &cost){
    int first = 0;
    int second = cost[0];
    for (int i=2; i<int(cost.size()); i++) {
        int tmp = second;
        second = min(first, second)+cost[i];
        first = tmp;
    }
    return (first>second ? second:first);
}

int Solution::waysToStep(int n){
    vector<int> res(n+1);
    if (n==1) {
        return 1;
    }
    if (n==2) {
        return 2;
    }
    if (n==3) {
        return 4;
    }
    res[0] = 0;
    res[1] = 1;
    res[2] = 2;
    res[3] = 4;
    for (int i=4; i<=n; i++) {
        res[i] = ((res[i-1]%1000000007+res[i-2]%1000000007)%1000000007+res[i-3])%1000000007;
    }
    return res[n];
}

vector<vector<int>> Solution::threeSum(vector<int> &nums){
    // 思想：还是使用遍历解决，避免出现重复元素是关键，将三数之和改为两数之和
    vector<int> sorted_nums = quicksort(nums);                          //首先进行排序
    vector<vector<int>> res;
    int length = int(nums.size());
    for (int i=0; i<length; i++) {                                      //i保存两数之和的目标值
        if (i>0 && sorted_nums[i]==sorted_nums[i-1]) {                  //如果说数组存在重复，则忽略
            continue;
        }
        int target = -sorted_nums[i];
        int k = length-1;                                               //k可以放在j的外层，因为数组有序
                                                                        //随着j的增大，k必然减小
        for (int j=i+1; j<length; j++) {
            if (j>i+1 && sorted_nums[j]==sorted_nums[j-1]) {            //避免重复
                continue;
            }
            while (j<k && sorted_nums[j]+sorted_nums[k]>target) {
                k--;
            }
            if (j==k) {
                break;
            }
            if (sorted_nums[j]+sorted_nums[k]==target) {
                res.push_back({sorted_nums[i],sorted_nums[j],sorted_nums[k]});
            }
        }
    }
    return res;
}

vector<int> Solution::quicksort(vector<int>& nums){
    if (nums.size()>=2) {
        int mid = nums[0];
        vector<int> left;
        vector<int> right;
        for (int i=1; i<int(nums.size()); i++) {
            if (nums[i]<mid) {
                left.push_back(nums[i]);
            }else{
                right.push_back(nums[i]);
            }
        }
        return quicksort_join(quicksort(left), mid, quicksort(right));
    }
    return nums;
}

vector<int> Solution::quicksort_join(const vector<int>& left, int mid, const vector<int>& right){
    vector<int> res;
    for (int n:left) {
        res.push_back(n);
    }
    res.push_back(mid);
    for (int n:right) {
        res.push_back(n);
    }
    return res;
}

int Solution::massage(vector<int> &nums){
    int length = int(nums.size());
    if (length==0) {
        return NULL;
    }
    if (length==1) {
        return nums[0];
    }
    int first = 0;
    int second = nums[0];
    for (int i=1; i<length; i++) {
        int tmp = second;
        second = max(nums[i]+first, second);
        first = tmp;
    }
    return second;
}

int Solution::maxProfit(vector<int> &prices){
    int length = int(prices.size());
    if(length==0){return NULL;}
    int mini = prices[0];
    int profit = 0;
    for (int i=1; i<length; i++) {
        mini = min(prices[i], mini);
        profit = (prices[i]-mini>profit ? prices[i]-mini:profit);
    }
    return profit;
}

vector<int> Solution::countBits(int num){
    vector<int> res(num+1);
    res[0] = 0;
    for (int i=1; i<=num; i++) {
        res[i] = res[i&(i-1)]+1;
    }
    return res;
}

int Solution::findBestValue(vector<int> &arr, int target){
    sort(arr.begin(), arr.end());                           //首先进行排序
    int length = int(arr.size());
    int sum = 0;
    for(int n:arr){                                         //求数组和
        sum += n;
    }
    // cout << "sum:" << sum << endl;
    int diff = abs(sum-target);                             //从后向前遍历，找到与目标值之差不再增大的索引
    int res = arr.back();
    int backward_sum = arr.back();
    int idx = length-1;
    for(int i=length-2; i>=0; i--){
        int sum_tmp = sum-backward_sum+arr[i]*(length-1-i);
        int diff_tmp = abs(sum_tmp-target);
        if (diff_tmp>diff) {
            break;
        }else{
            diff = diff_tmp;
            res = arr[i];
            backward_sum += arr[i];
            idx = i;
        }
    }
    //cout << "idx:" << idx << endl;
    //cout << arr.front() << endl;
    if (idx==0) {                                                   //如果说是直到找到第一个值并且大于平均，则只在两个值内取舍
        int res_tmp = target/arr.size();
        if (res_tmp<=arr.front()) {
            int first = abs(target-res_tmp*int(arr.size()));
            int second = abs((res_tmp+1)*int(arr.size())-target);
            if (first>second) {
                return res_tmp+1;
            }else{
                return res_tmp;
            }
        }else{                                          //否则在第一个值和第二个值之间进行寻找
            diff = sum-(arr[idx]+arr[idx+1]*(length-1));
            //cout << diff << endl;
            for (int i=arr[idx]+1; i<arr[idx+1]; i++) {
                int sum_tmp = sum-backward_sum+arr[idx]+i*(length-1-idx);
                int diff_tmp = abs(sum_tmp-target);
                if(diff_tmp<diff){
                    res = i;
                    diff = diff_tmp;
                }
            }
        }
    }else{                                              //如idx不为0，
        //cout << "idx:" << idx << endl;
        for (int i=arr[idx-1]+1; i<arr[idx]; i++) {
            //cout << "i:" << i << endl;
            // cout << backward_sum << endl;
            int sum_tmp = sum-backward_sum+i*(length-idx);
            int diff_tmp = abs(sum_tmp-target);
            if(diff_tmp<diff){
                res = i;
                diff = diff_tmp;
            }
        }
    }
    return res;
}

string Solution::longestCommonPrefix(vector<string> &strs){
    int length = int(strs.size());
    int mini = int(strs[0].size());
    string min_s = strs[0];
    for (int i=1; i<length; i++) {
        if (strs[i].size()<mini) {
            mini = int(strs[i].size());
            min_s = strs[i];
        }
    }
    int idx = -1;
    for (int i=0; i<mini; i++) {
        bool con = true;
        for (int j=0; j<length; j++) {
            if (strs[j][i]!=min_s[i]) {
                con = false;
                break;
            }
        }
        if (!con) {
            break;
        }
        idx = i;
    }
    string res = (idx==-1 ? "":min_s.substr(0,idx+1));
    return res;
}
/* java 代码。// 297. 二叉树的序列化与反序列化 用栈也可以实现二叉树的遍历,但是是层序遍历
 public class Codec {
     public String serialize(TreeNode root) {
         //tree: [v1,v2,null,...]
         StringBuilder res = new StringBuilder("[");
         Queue<TreeNode> queue = new LinkedList();
         queue.add(root);
         while(!queue.isEmpty()){
             TreeNode cur = queue.remove();
             if(cur == null){
                 res.append("null,");
             }else{
                 res.append(cur.val + ",");
                 queue.add(cur.left);
                 queue.add(cur.right);
             }
         }
         res.setLength(res.length() - 1);
         res.append("]");
         return res.toString();
     }

     public TreeNode deserialize(String data) {
         String[] nodes = data.substring(1, data.length()-1).split(",");
         TreeNode root = getNode(nodes[0]);
         Queue<TreeNode> parents = new LinkedList();
         TreeNode parent = root;
         boolean isLeft = true;
         for(int i = 1; i < nodes.length; i++){
             TreeNode cur = getNode(nodes[i]);
             if(isLeft){
                 parent.left = cur;
             }else{
                 parent.right = cur;
             }
             if(cur != null){
                 parents.add(cur);
             }
             isLeft = !isLeft;
             if(isLeft){
                 parent = parents.poll();
             }
         }
         return root;
     }

     private TreeNode getNode(String val){
         if(val.equals("null")){
             return null;
         }
         return new TreeNode(Integer.valueOf(val));
     }
 }
 */
string Codec::serialize(TreeNode *root){
    stack<TreeNode*> s;
    string res = "[";
    TreeNode *node = root;
    res += (to_string(node->val)+",");
    while (node!=NULL || !s.empty()) {
        while (node!=NULL) {
            s.push(node);
            node = node->left;
            if (node!=NULL) {
                res += (to_string(node->val)+",");
            }else{
                res += "null,";
            }
        }
        if (!s.empty()) {
            node = s.top();
            s.pop();
            node = node->right;
            if (node!=NULL) {
                res += (to_string(node->val)+",");
            }else{
                res += "null,";
            }
        }
    }
    res[res.size()-1] = ']';
    return res;
}

TreeNode* Codec::deserialize(string data){
    vector<string> list;
    string content = data.substr(1,data.size()-1);
    string ele = "";
    for(char n:content){
        if (n!=',' && n!=']') {
            ele += n;
        }else{
            list.push_back(ele);
            ele = "";
        }
    }
    
    return rdeserialize(list);
}

TreeNode* Codec::rdeserialize(vector<string> list){
    if (list[0]=="null") {
        list.erase(list.begin());
        return NULL;
    }
    TreeNode* root = new TreeNode(stoi(list[0]));
    list.erase(list.begin());
    root->left = rdeserialize(list);
    root->right = rdeserialize(list);
    return root;
}


/*
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 
class Codec {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
    stack<TreeNode*> s;
    string res = "[";
    TreeNode *node = root;
    res += (to_string(node->val)+",");
    while (node!=NULL || !s.empty()) {
        while (node!=NULL) {
            s.push(node);
            node = node->left;
            if (node!=NULL) {
                res += (to_string(node->val)+",");
            }else{
                res += "null,";
            }
        }
        if (!s.empty()) {
            node = s.top();
            s.pop();
            node = node->right;
            if (node!=NULL) {
                res += (to_string(node->val)+",");
            }else{
                res += "null,";
            }
        }
    }
    res[res.size()-1] = ']';
    return res;
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
    vector<string> list;
    string content = data.substr(1,data.size()-1);
    string ele = "";
    for(char n:content){
        if (n!=',' && n!=']') {
            ele += n;
        }else{
            list.push_back(ele);
            ele = "";
        }
    }
    
    return rdeserialize(list);
    }
    TreeNode* rdeserialize(vector<string> list){
        if (list[0]=="null") {
            list.erase(list.begin());
            return NULL;
        }
        TreeNode* root = new TreeNode(stoi(list[0]));
        list.erase(list.begin());
        root->left = rdeserialize(list);
        root->right = rdeserialize(list);
        return root;
    }
};

// Your Codec object will be instantiated and called as such:
// Codec codec;
// codec.deserialize(codec.serialize(root));
*/

TreeNode* Codec::preorder_recursion(TreeNode *root){
    if (!root) {
        return NULL;
    }
    cout << root->val << endl;
    preorder_recursion(root->left);
    preorder_recursion(root->right);
    return NULL;
}

TreeNode* Solution::sortedArrayToBST(vector<int> &nums){
    return sortedArrayToBST_dfs(0, int(nums.size())-1, nums);
}
TreeNode* Solution::sortedArrayToBST_dfs(int l, int r, vector<int> nums){
    if (l>r) {
        return NULL;
    }
    int idx = (r+l+1)/2;                            /* 通过均分的方法达到平衡的目的 */
    int x = nums[idx];
    TreeNode* root = new TreeNode(x);
    root->left = sortedArrayToBST_dfs(l, idx-1, nums);
    root->right = sortedArrayToBST_dfs(idx+1, r, nums);
    return root;
}

int Solution::maxDepth(TreeNode *root){             /* 有点类似于动态规划 */
    if (!root) {
        return 0;
    }
    return max(maxDepth(root->left),maxDepth(root->right))+1;
}

TreeNode* Solution::mirrorTree(TreeNode *root){     /* 传入root，返回root */
    if (!root) {                                    /* 中间细节不必考虑太过清楚 */
        return NULL;
    }
    TreeNode* tmp = root->left;
    root->left = mirrorTree(root->right);
    root->right = mirrorTree(tmp);
    return root;
}

int Solution::rangeSumBST(TreeNode *root, int L, int R){
    if (!root) {
        return NULL;
    }
    int sum = 0;
    stack<TreeNode*> s;
    s.push(root);
    while (!s.empty()) {
        TreeNode* node = s.top();
        s.pop();
        if (L<=node->val && node->val<=R) {
            sum += node->val;
            if (node->right) {s.push(node->right);}
            if (node->left) {s.push(node->left);}
        }
        if (node->val<L) {
            if (node->right) {s.push(node->right);}
        }
        if (node->val>R) {
            if (node->left) {s.push(node->left);}
        }
    }
    return sum;
}

TreeNode* Solution::mergeTrees(TreeNode *t1, TreeNode *t2){
    if (!t1 && !t2) {
        return NULL;
    }
    if (t1 && t2) {
        TreeNode* node = new TreeNode(t1->val+t2->val);
        node->left = mergeTrees(t1->left, t2->left);
        node->right = mergeTrees(t1->right, t2->right);
        return node;
    }else if(t1){
        return t1;
    }else{
        return t2;
    }
}

TreeNode* Solution::searchBST(TreeNode *root, int val){
    stack<TreeNode*> s;
    s.push(root);
    while (!s.empty()) {
        TreeNode* node = s.top();
        s.pop();
        if (node->val==val) {
            return node;
        }
        if (node->right) {s.push(node->right);}
        if (node->left) {s.push(node->left);}
    }
    return NULL;
}

vector<int> Solution::postorder(NNode *root){
    //[children of v1], v1, [children of v2], v2, [children of v3], v3, u
    //u, v3, [children of v3]', v2, [children of v2]', v1, [children of v1]'
    //后序遍历和前序遍历类似，只不过子节点的顺序是反的
    vector<int> res;
    if (!root) {
        return res;
    }
    stack<NNode*> s;
    s.push(root);
    while (!s.empty()) {
        NNode* node = s.top();
        s.pop();
        int len = int(node->children.size());
        cout << node->val << endl;
        res.push_back(node->val);
        for (int i=0; i<len; i++) {
            s.push(node->children[i]);
        }
    }
    reverse(res.begin(), res.end());
    return res;
}

vector<int> Solution::preorder(NNode *root){
    vector<int> res;
    stack<NNode*> s;
    while (true) {
        preorder_visit(root, s, res);
        if (s.empty()) {break;}
        root = s.top();
        s.pop();
    }
    return res;
}

void Solution::preorder_visit(NNode *root, stack<NNode *>& s, vector<int>& res){
    while (root) {
        res.push_back(root->val);
        if (root->children.size()>1) {
            for (int i=int(root->children.size())-1; i>0; i--) {
                s.push(root->children[i]);
            }
            root = root->children[0];
        }else if(root->children.size()){
            root = root->children[0];
        }else{
            root = NULL;
        }
    }
}

int Solution::kthLargest(TreeNode *root, int k){
    stack<TreeNode*> s;
    vector<int> res;
    while (true) {
        kthLargest_visit(root, s);
        if (s.empty()) {break;}
        root = s.top();
        s.pop();
        res.push_back(root->val);
        root = root->right;
    }
    return res[res.size()-k];
}

void Solution::kthLargest_visit(TreeNode *root, stack<TreeNode*> &s){
    while (root) {
        s.push(root);
        root = root->left;
    }
}

int Solution::maxDepthN(NNode *root){
    if (!root) {
        return 0;
    }
    vector<int> res;
    if (root->children.size()) {
        for(auto node:root->children){
            res.push_back(maxDepthN(node)+1);
        }
    }else{
        res.push_back(1);
    }
    sort(res.begin(),res.end());
    return res[res.size()-1];
}

TreeNode* Solution::increasingBST(TreeNode *root){
    stack<TreeNode*> s;
    vector<int> res;
    while (true) {
        increasingBST_visit(s, root);
        if (s.empty()) {break;}
        root = s.top();
        s.pop();
        res.push_back(root->val);
        root = root->right;
    }
    root = new TreeNode(res[0]);
    TreeNode* R = root;
    for (int i=1; i<res.size(); i++) {
        TreeNode* node = new TreeNode(res[i]);
        root->right = node;
        root = node;
    }
    return R;
}

void Solution::increasingBST_visit(stack<TreeNode *> &s, TreeNode *root){
    while (root) {
        s.push(root);
        root = root->left;
    }
}
