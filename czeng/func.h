#include <pthread.h> // P-Threads Library Headers
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <math.h>
#include <string.h>
#include <iostream>
using namespace std;

#define ROUND 50
#define TRACE 1000
#define NUM_PACKET 1000*1000

class Set{
	public:
		int size;
		int * element;
		Set(){
			size = 0;
			element = new int;
		}
};

class hash_slot{
public:  
    long key; 
    Set *set;
    hash_slot(){
        key = -1 ;
        set = NULL ;
    }
};

struct Range {
    long lb;
    long ub;
    long * BV;
} ;

struct TreeNode{
		TreeNode * left;
		TreeNode * right;
		long ub;
		long lb;
		long * BV;
};

struct Point{     
		long value;
		Set  *set;
};

struct thread_param_t{  
		int thread_id;
		long * TreeArray;
		int TreeSize;
		long *input;
		Set * output;
		Set * Lists;
};

struct thread_param_t_H{  
    hash_slot * Table;   
    int hash_function_key;
    long * trace;   
		Set * output;
};

struct merge_thread_param_t{     
		Set * Set1;
		Set * Set2;
		Set * Set3;
		Set * Set4;
		Set * Set5;
		Set * Set6;
		Set * Set7;
		Set * Set8;
		Set * Set9;
};

long * generate_seeds (int * Seed_size, int bit_length);
long * generate_rules (long * seed, int seed_size, int rule_size, int bit_length);
void lb_ub (long *lb, long *ub, int size);
void cout_tree(TreeNode * node);
void quickSort(long *array,int left,int right);
int trim_bound (long *array,int left,int right);
void init_points(Point * points, long * bounds, int num);
TreeNode* RangesToBST(Range* ranges, int start, int end);
Point * ConstructTree_points(long *seed , int seed_size, long * rule, int rule_size);
void cout_BV (int * BV, int NUM_RULES);
void Generate_BV(void* foo);
void Lookup(void* foo);
void Lookup_Hash(void* foo);
void Merge(void* foo);
long find_empty_slot (Point * points, int * point_index, char method, hash_slot * cuckoo, int function_key, char * function, int * index, int * kick_out_method);
int generate_hash_table (hash_slot * cuckoo, int hash_size, Point * points, int point_size, int function_key);
void Gen_tree_2 (long *array, int index, int array_size, long * bounds, int start, int end);