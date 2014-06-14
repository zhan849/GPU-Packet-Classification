#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cublas.h>
#define field 15
#define rule 1000
using namespace std;

__host__ int** tree_gen(int field, int rule){
	srand(time(NULL));
	int** tree = new int*[field];
	for (int i = 0; i < field; i++){
		tree[i] = new int[rule];
		for (int j = 0; j < rule; j++){
			tree[i][j] = rand()%100;
		}
	}
	return tree;
}
__host__ int** headers_gen(int num, int field){
	int** headers = new int*[field];
	for (int i = 0; i < field; i++){
		headers[i] = new int[num];
		for (int j = 0; j < num; j++){
			headers[i][j] = rand()%100;
		}
	}
	return headers;
}
__global__ void packet_classify( int* gpu_tree1, int* gpu_tree2, int* gpu_tree3, int* gpu_tree4, int* gpu_tree5,
								 int* gpu_tree6, int* gpu_tree7, int* gpu_tree8, int* gpu_tree9, int* gpu_tree10,
								 int* gpu_tree11, int* gpu_tree12, int* gpu_tree13, int* gpu_tree14, int* gpu_tree15,

								 int* gpu_field1, int* gpu_field2, int* gpu_field3, int* gpu_field4, int* gpu_field5, 
								 int* gpu_field6, int* gpu_field7, int* gpu_field8, int* gpu_field9, int* gpu_field10,
								 int* gpu_field11, int* gpu_field12, int* gpu_field13, int* gpu_field14, int* gpu_field15,

								 unsigned long long int* gpu_final_bv, int rule, int field, int packet_num      	){

	__shared__ int tree_shared[15][1000], bv_shared[15][1000];

/*
	__shared__ int* tree1_shared = new int[rule];
	__shared__ int* tree2_shared = new int[rule];
	__shared__ int* tree3_shared = new int[rule];
	__shared__ int* tree4_shared = new int[rule];
	__shared__ int* tree5_shared = new int[rule];
	__shared__ int* tree6_shared = new int[rule];
	__shared__ int* tree7_shared = new int[rule];
	__shared__ int* tree8_shared = new int[rule];
	__shared__ int* tree9_shared = new int[rule];
	__shared__ int* tree10_shared = new int[rule];
	__shared__ int* tree11_shared = new int[rule];
	__shared__ int* tree12_shared = new int[rule];
	__shared__ int* tree13_shared = new int[rule];
	__shared__ int* tree14_shared = new int[rule];
	__shared__ int* tree15_shared = new int[rule];

	__shared__ unsigned long long int* bv1_shared = new unsigned long long int[rule];
	__shared__ unsigned long long int* bv2_shared = new unsigned long long int[rule];
	__shared__ unsigned long long int* bv3_shared = new unsigned long long int[rule];
	__shared__ unsigned long long int* bv4_shared = new unsigned long long int[rule];
	__shared__ unsigned long long int* bv5_shared = new unsigned long long int[rule];
	__shared__ unsigned long long int* bv6_shared = new unsigned long long int[rule];
	__shared__ unsigned long long int* bv7_shared = new unsigned long long int[rule];
	__shared__ unsigned long long int* bv8_shared = new unsigned long long int[rule];
	__shared__ unsigned long long int* bv9_shared = new unsigned long long int[rule];
	__shared__ unsigned long long int* bv10_shared = new unsigned long long int[rule];
	__shared__ unsigned long long int* bv11_shared = new unsigned long long int[rule];
	__shared__ unsigned long long int* bv12_shared = new unsigned long long int[rule];
	__shared__ unsigned long long int* bv13_shared = new unsigned long long int[rule];
	__shared__ unsigned long long int* bv14_shared = new unsigned long long int[rule];
	__shared__ unsigned long long int* bv15_shared = new unsigned long long int[rule];

	__shared__ unsigned long long int** bv_all = new unsigned long long int*[15];

	// Copy the tree and the bit vectors into shared memory
	for (int i = 0; i < rule; i++){
		tree1_shared[i] = gpu_tree1[i];
		bv1_shared[i] = gpu_bv1[i];

		tree2_shared[i] = gpu_tree2[i];
		bv2_shared[i] = gpu_bv2[i];
		
		tree3_shared[i] = gpu_tree3[i];
		bv3_shared[i] = gpu_bv3[i];
		
		tree4_shared[i] = gpu_tree4[i];
		bv4_shared[i] = gpu_bv4[i];
		
		tree5_shared[i] = gpu_tree5[i];
		bv5_shared[i] = gpu_bv5[i];
		
		tree6_shared[i] = gpu_tree6[i];
		bv6_shared[i] = gpu_bv6[i];
		
		tree7_shared[i] = gpu_tree7[i];
		bv7_shared[i] = gpu_bv7[i];
		
		tree8_shared[i] = gpu_tree8[i];
		bv8_shared[i] = gpu_bv8[i];
		
		tree9_shared[i] = gpu_tree9[i];
		bv9_shared[i] = gpu_bv9[i];

		tree10_shared[i] = gpu_tree10[i];
		bv10_shared[i] = gpu_bv10[i];
		
		tree11_shared[i] = gpu_tree11[i];
		bv11_shared[i] = gpu_bv11[i];

		tree12_shared[i] = gpu_tree12[i];
		bv12_shared[i] = gpu_bv12[i];
		
		tree13_shared[i] = gpu_tree13[i];
		bv13_shared[i] = gpu_bv13[i];
		
		tree14_shared[i] = gpu_tree14[i];
		bv14_shared[i] = gpu_bv14[i];
		
		tree15_shared[i] = gpu_tree15[i];
		bv15_shared[i] = gpu_bv15[i];
	}
	*/
	/*
	__syncthreads();

	// Start packet classification
	if ((blockIDx.x * blockDim.x + threadIdx.x/15) > packet_num){
		return;
	}

	int i=0, value=tree1_shared[i];
	while (i < rules){
		if (gpu_field1[blockDim.x * blockIdx.x + threadIdx.x/15] == tree1_shared[i]){
			local_bv = i;
			break;
		}

	}
	*/



}
int main(int argc, char** argv){
	if(argc!=4){
		cout<<"usage ./openflow  *Packet_num   *Grid_dim   *Block_dim"<<endl; 
		return 0;
	}
	int packet_num = atoi(argv[1]);
	int grid_dim = atoi(argv[2]);
	int block_dim = atoi(argv[3]);
	cout<<"grid_dim: "<<grid_dim<<", block_dim: "<<block_dim<<", packet_num: "<<packet_num<<endl;
	
// Generating random packet headers and routing tables
	int** tree = tree_gen(field, rule);
	int** headers = headers_gen(packet_num, field);
	unsigned long long int* BV1 = new unsigned long long int[rule];
	unsigned long long int* BV2 = new unsigned long long int[rule];
	unsigned long long int* BV3 = new unsigned long long int[rule];
	unsigned long long int* BV4 = new unsigned long long int[rule];
	unsigned long long int* BV5 = new unsigned long long int[rule];
	unsigned long long int* BV6 = new unsigned long long int[rule];
	unsigned long long int* BV7 = new unsigned long long int[rule];
	unsigned long long int* BV8 = new unsigned long long int[rule];
	unsigned long long int* BV9 = new unsigned long long int[rule];
	unsigned long long int* BV10 = new unsigned long long int[rule];
	unsigned long long int* BV11 = new unsigned long long int[rule];
	unsigned long long int* BV12 = new unsigned long long int[rule];
	unsigned long long int* BV13 = new unsigned long long int[rule];
	unsigned long long int* BV14 = new unsigned long long int[rule];
	unsigned long long int* BV15 = new unsigned long long int[rule];
// Test packet headers and routing tables
	cout<<"Tree: "<<endl;
	for (int i = 0; i < rule; i++){
		for (int j = 0; j < field; j++){
			cout<<tree[i][j]<<" ";
		}
		cout<<endl;
	}
	
	cout<<endl<<"Headers: "<<endl;
	for (int i = 0; i < packet_num; i++){
		for (int j = 0; j < field; j++){
			cout<<headers[i][j]<<" ";
		}
		cout<<endl;
	}

// Define classification variables
	dim3 dimGrid(grid_dim,1);
	dim3 dimBlock(block_dim,1);
	/*
	int *gpu_tree1, *gpu_tree2, *gpu_tree3, *gpu_tree4, *gpu_tree5, 
		*gpu_tree6, *gpu_tree7, *gpu_tree8, *gpu_tree9, *gpu_tree10, 
		*gpu_tree11, *gpu_tree12, *gpu_tree13, *gpu_tree14, *gpu_tree15; 

	int *gpu_field1, *gpu_field2, *gpu_field3, *gpu_field4, *gpu_field5, 
		*gpu_field6, *gpu_field7, *gpu_field8, *gpu_field9, *gpu_field10, 
		*gpu_field11, *gpu_field12, *gpu_field13, *gpu_field14, *gpu_field15;

	unsigned long long int *gpu_bv1, *gpu_bv2, *gpu_bv3, *gpu_bv4, *gpu_bv5, 
		*gpu_bv6, *gpu_bv7, *gpu_bv8, *gpu_bv9, *gpu_bv10, 
		*gpu_bv11, *gpu_bv12, *gpu_bv13, *gpu_bv14, *gpu_bv15, *gpu_final_bv;
*/
// Define timer variables
		float time1, time2;
		cudaEvent_t start2, stop2, start1, stop1;

// Allocate memories for headers and trees
		// Trees
		cudaMalloc((void**)&gpu_tree1, sizeof(int)*rule); 
		cudaMalloc((void**)&gpu_tree2, sizeof(int)*rule);
		cudaMalloc((void**)&gpu_tree3, sizeof(int)*rule); 
		cudaMalloc((void**)&gpu_tree4, sizeof(int)*rule);
		cudaMalloc((void**)&gpu_tree5, sizeof(int)*rule); 
		cudaMalloc((void**)&gpu_tree6, sizeof(int)*rule);
		cudaMalloc((void**)&gpu_tree7, sizeof(int)*rule); 
		cudaMalloc((void**)&gpu_tree8, sizeof(int)*rule);
		cudaMalloc((void**)&gpu_tree9, sizeof(int)*rule); 
		cudaMalloc((void**)&gpu_tree10, sizeof(int)*rule);
		cudaMalloc((void**)&gpu_tree11, sizeof(int)*rule); 
		cudaMalloc((void**)&gpu_tree12, sizeof(int)*rule);
		cudaMalloc((void**)&gpu_tree13, sizeof(int)*rule); 
		cudaMalloc((void**)&gpu_tree14, sizeof(int)*rule);
		cudaMalloc((void**)&gpu_tree15, sizeof(int)*rule);
		// Fields 
		cudaMalloc((void**)&gpu_field1, sizeof(int)*packet_num);
		cudaMalloc((void**)&gpu_field2, sizeof(int)*packet_num);
		cudaMalloc((void**)&gpu_field3, sizeof(int)*packet_num);
		cudaMalloc((void**)&gpu_field4, sizeof(int)*packet_num);
		cudaMalloc((void**)&gpu_field5, sizeof(int)*packet_num);
		cudaMalloc((void**)&gpu_field6, sizeof(int)*packet_num);
		cudaMalloc((void**)&gpu_field7, sizeof(int)*packet_num);
		cudaMalloc((void**)&gpu_field8, sizeof(int)*packet_num);
		cudaMalloc((void**)&gpu_field9, sizeof(int)*packet_num);
		cudaMalloc((void**)&gpu_field10, sizeof(int)*packet_num);
		cudaMalloc((void**)&gpu_field11, sizeof(int)*packet_num);
		cudaMalloc((void**)&gpu_field12, sizeof(int)*packet_num);
		cudaMalloc((void**)&gpu_field13, sizeof(int)*packet_num);
		cudaMalloc((void**)&gpu_field14, sizeof(int)*packet_num);
		cudaMalloc((void**)&gpu_field15, sizeof(int)*packet_num);
		// BVs
		cudaMalloc((void**)&gpu_bv1, sizeof(unsigned long long int)*field);
		cudaMalloc((void**)&gpu_bv2, sizeof(unsigned long long int)*field);
		cudaMalloc((void**)&gpu_bv3, sizeof(unsigned long long int)*field);
		cudaMalloc((void**)&gpu_bv4, sizeof(unsigned long long int)*field);
		cudaMalloc((void**)&gpu_bv5, sizeof(unsigned long long int)*field);
		cudaMalloc((void**)&gpu_bv6, sizeof(unsigned long long int)*field);
		cudaMalloc((void**)&gpu_bv7, sizeof(unsigned long long int)*field);
		cudaMalloc((void**)&gpu_bv8, sizeof(unsigned long long int)*field);
		cudaMalloc((void**)&gpu_bv9, sizeof(unsigned long long int)*field);
		cudaMalloc((void**)&gpu_bv10, sizeof(unsigned long long int)*field);
		cudaMalloc((void**)&gpu_bv11, sizeof(unsigned long long int)*field);
		cudaMalloc((void**)&gpu_bv12, sizeof(unsigned long long int)*field);
		cudaMalloc((void**)&gpu_bv13, sizeof(unsigned long long int)*field);
		cudaMalloc((void**)&gpu_bv14, sizeof(unsigned long long int)*field);
		cudaMalloc((void**)&gpu_bv15, sizeof(unsigned long long int)*field);
		cudaMalloc((void**)&gpu_final_bv, sizeof(unsigned long long int)*packet_num);
		// Timers
		cudaEventCreate(&start1);
		cudaEventCreate(&start2);
		cudaEventCreate(&stop1);
		cudaEventCreate(&stop2);

/* Place for time recording #1 */

// Memory copy
		cudaMemcpy(gpu_tree1, tree[0], sizeof(int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_tree2, tree[1], sizeof(int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_tree3, tree[2], sizeof(int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_tree4, tree[3], sizeof(int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_tree5, tree[4], sizeof(int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_tree6, tree[5], sizeof(int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_tree7, tree[6], sizeof(int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_tree8, tree[7], sizeof(int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_tree9, tree[8], sizeof(int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_tree10, tree[9], sizeof(int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_tree11, tree[10], sizeof(int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_tree12, tree[11], sizeof(int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_tree13, tree[12], sizeof(int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_tree14, tree[13], sizeof(int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_tree15, tree[14], sizeof(int)*rule, cudaMemcpyHostToDevice);

		cudaMemcpy(gpu_field1, headers[0], sizeof(int)*packet_num, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_field2, headers[1], sizeof(int)*packet_num, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_field3, headers[2], sizeof(int)*packet_num, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_field4, headers[3], sizeof(int)*packet_num, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_field5, headers[4], sizeof(int)*packet_num, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_field6, headers[5], sizeof(int)*packet_num, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_field7, headers[6], sizeof(int)*packet_num, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_field8, headers[7], sizeof(int)*packet_num, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_field9, headers[8], sizeof(int)*packet_num, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_field10, headers[9], sizeof(int)*packet_num, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_field11, headers[10], sizeof(int)*packet_num, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_field12, headers[11], sizeof(int)*packet_num, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_field13, headers[12], sizeof(int)*packet_num, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_field14, headers[13], sizeof(int)*packet_num, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_field15, headers[14], sizeof(int)*packet_num, cudaMemcpyHostToDevice);

		cudaMemcpy(gpu_bv1, BV1, sizeof(unsigned long long int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_bv2, BV2, sizeof(unsigned long long int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_bv3, BV3, sizeof(unsigned long long int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_bv4, BV4, sizeof(unsigned long long int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_bv5, BV5, sizeof(unsigned long long int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_bv6, BV6, sizeof(unsigned long long int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_bv7, BV7, sizeof(unsigned long long int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_bv8, BV8, sizeof(unsigned long long int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_bv9, BV9, sizeof(unsigned long long int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_bv10, BV10, sizeof(unsigned long long int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_bv11, BV11, sizeof(unsigned long long int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_bv12, BV12, sizeof(unsigned long long int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_bv13, BV13, sizeof(unsigned long long int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_bv14, BV14, sizeof(unsigned long long int)*rule, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_bv15, BV15, sizeof(unsigned long long int)*rule, cudaMemcpyHostToDevice);

/* Place for time recording #2 */














	for (int i = 0; i < field; i++){
		delete tree[i];
		delete headers[i];
	} 
	delete tree;
	delete headers;

	return 0;
}