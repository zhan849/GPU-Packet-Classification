#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cublas.h>

using namespace std;

#define FIELD 15
#define RULE 511
#define ALLRULE 2048
#define WSIZE 32
#define int_count ALLRULE / (sizeof(long int) * 8)

struct Stage1_Data{
	int* tree;
	int* header;
	int packet_num;
	int block_dim;
};

struct Stage2_Data{
	long int* bv;
	int* match_result;
	long int* merge_result_partial;
	long int* bv_final;
};
struct pthread_param_P{
	int thread_id;
	long int merge_source[FIELD];
	long int* merge_result_partial;
};
struct pthread_param_F{
	int thread_id;
	long int* bv_final;
	long int merge_source[int_count];
};
struct pthread_param_C{
	int thread_id;
	//long int* bv_final;
	//long int partial_merge_source[int_count * FIELD];
	long int* partial_merge_source[FIELD];
	long int merge_result[int_count];
};
void tree_gen(int** tree, int field, int rule);
void header_gen(int** headers, int** tree, int field, int packet_num);
void bv_gen(long int** bv, long int* bv_final, int packet_num);
void data_test(int** tree, int** headers, long int** bv, int* bv_final, int packet_num, int type);
__global__ void packet_classify(int* gpu_tree, int* gpu_headers, int* gpu_match_result, int packet_num, int block_dim);
__global__ void packet_merge(long int* gpu_bv, int* gpu_match_result, long int* gpu_merge_result, long int*gpu_bv_final, int packet_num, int block_dim);
void partial_merge(void* foo);
void final_merge(void* foo);
void merge(void* foo);






