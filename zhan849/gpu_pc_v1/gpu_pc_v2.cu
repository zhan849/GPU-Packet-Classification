/********************************************************
*
*   This experiment optimizes packet classification
*   in the following aspects:
*     1. Thread assignment
*     2. Memory coalescing
*	
*	Experiment Assumptions:
*		1. 510 Non-overlapping intervals
*		2. 1024 Rules (510 * 1024 element BVs)
*		3. Number of packets varies, 1 kernel
*		4. All packets are already on CPU memory
*		5. All fields needs prefix/range match
*
********************************************************/


#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cublas.h>

#define FIELD 6
#define RULE 31
#define ALLRULE 128
#define WSIZE 32
#define int_count ALLRULE/sizeof(long int)

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

using namespace std;

void header_gen(int**, int**, int, int);
void tree_gen(int**, int, int);
void bv_gen(long int**, long int*, int);
void data_test(int**, int**, long int**, int*, int, int);

__global__ void packet_classify(int* gpu_tree, int* gpu_headers, long int* gpu_bv, int* gpu_match_result, long int* gpu_merge_result, int packet_num, int block_dim){
	__shared__ int gpu_tree_shared[FIELD*RULE];
	int level = 0;
	while(level * block_dim + threadIdx.x < FIELD * RULE){
		gpu_tree_shared[level * block_dim + threadIdx.x] = gpu_tree[level * block_dim + threadIdx.x];
		level++;
	}
	__syncthreads();

	//int index = blockDim.x * blockIdx.x + threadIdx.x;
	//int tree_idx = index / packet_num * RULE;
	int i = 0;
	if (blockDim.x * blockIdx.x + threadIdx.x < packet_num * FIELD){
		while (i < RULE){
			
			i = 2 * i + (gpu_headers[blockDim.x * blockIdx.x + threadIdx.x] <= gpu_tree_shared[(blockDim.x * blockIdx.x + threadIdx.x) / packet_num * RULE+i]) * 1 + (gpu_headers[blockDim.x * blockIdx.x + threadIdx.x] > gpu_tree_shared[(blockDim.x * blockIdx.x + threadIdx.x) / packet_num * RULE+i]) * 2;
		}
		gpu_match_result[blockDim.x * blockIdx.x + threadIdx.x] = i - RULE;
	}

	__syncthreads();
	
	if (blockDim.x * blockIdx.x + threadIdx.x < packet_num * int_count){
		int index = blockDim.x * blockIdx.x + threadIdx.x;
		int packetIdx = index/int_count;
		gpu_merge_result[index] = gpu_bv[gpu_match_result[packetIdx*FIELD]*int_count + index%int_count] &
								  gpu_bv[gpu_match_result[packetIdx*FIELD+1]*int_count + index%int_count];
	/*						  gpu_bv[gpu_match_result[packetIdx*15+2]*int_count + index%int_count] &
							  gpu_bv[gpu_match_result[packetIdx*15+3]*int_count + index%int_count] &
							  gpu_bv[gpu_match_result[packetIdx*15+4]*int_count + index%int_count] &
							  gpu_bv[gpu_match_result[packetIdx*15+5]*int_count + index%int_count] &
							  gpu_bv[gpu_match_result[packetIdx*15+6]*int_count + index%int_count] &
							  gpu_bv[gpu_match_result[packetIdx*15+7]*int_count + index%int_count] &
							  gpu_bv[gpu_match_result[packetIdx*15+8]*int_count + index%int_count] &
							  gpu_bv[gpu_match_result[packetIdx*15+9]*int_count + index%int_count] &
							  gpu_bv[gpu_match_result[packetIdx*15+10]*int_count + index%int_count] &
							  gpu_bv[gpu_match_result[packetIdx*15+11]*int_count + index%int_count] &
							  gpu_bv[gpu_match_result[packetIdx*15+12]*int_count + index%int_count] &
							  gpu_bv[gpu_match_result[packetIdx*15+13]*int_count + index%int_count] &
							  gpu_bv[gpu_match_result[packetIdx*15+14]*int_count + index%int_count];
*/
	}
};


int main(int argc, char** argv){
	if(argc!=4){
		cout<<"usage ./openflow  *Packet_num   *Grid_dim   *Block_dim"<<endl; 
		return 0;
	}
	int packet_num = atoi(argv[1]);
	int grid_dim = atoi(argv[2]);
	int block_dim = atoi(argv[3]);
//	if (grid_dim*block_dim != packet_num*FIELD){
//		cout<<"ERROR: Total number of threads must equal packet_num * FIELD"<<endl;
//		return 1;
//	}
	cout<<"grid_dim: "<<grid_dim<<", block_dim: "<<block_dim<<", packet_num: "<<packet_num<<endl;
cout<<"============================ Experiment Starts ============================"<<endl;
/********************************************************
*	Preparing Data:
*		1. Generate random headers 
*		2. Generate BVs 
*		3. Generate random packets
*		4. Deliberately make some rule-matching packets
********************************************************/
	srand(time(NULL));
	int** tree = new int*[FIELD];
		for(int i = 0; i < FIELD; i++){
			tree[i] = new int[RULE];
		}
	int** headers = new int*[FIELD];
		for (int i = 0; i < FIELD; i++){
			headers[i] = new int[packet_num];
		}
	long int** bv = new long int*[FIELD*(RULE+1)];
		for(int i = 0; i < FIELD*(RULE+1); i++){
			bv[i] = new long int[ALLRULE / sizeof(long int)];
		}
	long int* bv_final = new long int[packet_num * int_count];
	int* match_result = new int[packet_num * FIELD];
	long int* merge_results = new long int[int_count*packet_num];

	tree_gen(tree, FIELD, RULE);
	header_gen(headers, tree, FIELD, packet_num);
	bv_gen(bv, bv_final, packet_num);
	
	//data_test(tree, headers, bv, bv_final, packet_num, 3);

/********************************************************
*	Flatten All the 2D Arrays
********************************************************/
	int* tree_flatten = new int[RULE*FIELD];
	int* headers_flatten = new int[packet_num*FIELD];
	long int* bv_flatten = new long int[FIELD*(RULE+1) * ALLRULE / sizeof(long int)];

	for (int i = 0; i < FIELD; i++){
		for (int j = 0; j < RULE; j++){
			tree_flatten[i*RULE+j] = tree[i][j];
		}
	}
	for (int i = 0; i < FIELD; i++){
		for (int j = 0; j < packet_num; j++){
			headers_flatten[i*packet_num + j] = headers[i][j];
		}
	}
	for (int i = 0; i < FIELD*(RULE+1); i++){
		for (int j = 0; j < ALLRULE / sizeof(long int); j++){
			bv_flatten[i*ALLRULE / sizeof(long int) + j] = bv[i][j];
		}
	}
/********************************************************
*	Declare cuda events for statistical purposes:
*		1. time_memcpyH2D
*		2. time_memcpyD2H
*		3. time_pc
********************************************************/
	float time1, time2, time3;
	cudaEvent_t time_memcpyH2D_start, time_memcpyH2D_stop, time_memcpyD2H_start, time_memcpyD2H_stop, time_comp_start, time_comp_stop;
	cudaEventCreate(&time_memcpyH2D_start);
	cudaEventCreate(&time_memcpyH2D_stop);
	cudaEventCreate(&time_memcpyD2H_start);
	cudaEventCreate(&time_memcpyD2H_stop);
	cudaEventCreate(&time_comp_start);
	cudaEventCreate(&time_comp_stop);


/********************************************************
*	Allocate Space in Device:
*		1. gpu_tree 
*		2. gpu_bv 
*		3. gpu_bv_final
*		4. gpu_headers
********************************************************/
	dim3 dimGrid(grid_dim,1);
	dim3 dimBlock(block_dim,1);
	int* gpu_tree;
	int* gpu_headers;
	int* gpu_match_result;
	long int* gpu_merge_results;
	long int* gpu_bv;

	cudaMalloc((void**)&gpu_tree, sizeof(int*)*size_t(FIELD*RULE));
		cudaCheckErrors("cudaMalloc gpu_tree");
	cudaMalloc((void**)&gpu_headers, sizeof(int)*FIELD*packet_num);
		cudaCheckErrors("cudaMalloc gpu_headers");
	cudaMalloc((void**)&gpu_bv, sizeof(long int) * FIELD*(RULE+1) * int_count);
		cudaCheckErrors("cudaMalloc gpu_bv");
	cudaMalloc((void**)&gpu_match_result, sizeof(int)*packet_num*FIELD);
		cudaCheckErrors("cudaMalloc gpu_match_result");
	cudaMalloc((void**)&gpu_merge_results, sizeof(long int)*packet_num*int_count);
		cudaCheckErrors("cudaMalloc gpu_merge_results");

	cudaEventRecord(time_memcpyH2D_start, 0);
	
	cudaMemcpy(gpu_tree, tree_flatten, sizeof(int)*RULE*FIELD, cudaMemcpyHostToDevice);
		cudaCheckErrors("cudaMemcpy gpu_tree");
	cudaMemcpy(gpu_headers, headers_flatten, sizeof(int)*FIELD*packet_num, cudaMemcpyHostToDevice);
		cudaCheckErrors("cudaMemcpy gpu_headers");
	cudaMemcpy(gpu_bv, bv_flatten, sizeof(long int) * FIELD*(RULE+1) * int_count, cudaMemcpyHostToDevice);
		cudaCheckErrors("cudaMemcpy gpu_bv");

	cudaEventRecord(time_memcpyH2D_stop, 0);
	cudaEventSynchronize(time_memcpyH2D_stop);
	cudaEventElapsedTime(&time1, time_memcpyH2D_start, time_memcpyH2D_stop);
	cudaEventDestroy(time_memcpyH2D_stop);
	cudaEventDestroy(time_memcpyH2D_start);

	cout<<endl<<"*	1. Time for memcpy H2D: "<<time1<<"ms, Total bytes copied: "<<endl;
	cout<<"    		-> Tree: "<< sizeof(int)*RULE*FIELD<<" Bytes"<<endl;
	cout<<"    		-> Headers: "<< sizeof(long int)*FIELD*packet_num<<" Bytes"<<endl;
	cout<<"    		-> Bv: "<< sizeof(long int) * FIELD*(RULE+1) * int_count<<" Bytes"<<endl;
	cout<<"    		-> Total Memory Copy: "<< sizeof(int)*RULE*FIELD + sizeof(long int)*FIELD*packet_num + sizeof(long int) * FIELD*(RULE+1) * int_count<<" Bytes"<<endl;



/********************************************************
*	Main Packet Classification Process:
*		1. Function Call
*		2. Timing
*		3. Memory copy back (gpu_bv_final)
********************************************************/

	cudaEventRecord(time_comp_start, 0);

	packet_classify<<<dimGrid, dimBlock>>>(gpu_tree, gpu_headers, gpu_bv, gpu_match_result, gpu_merge_results, packet_num, block_dim);
	cudaCheckErrors("Kernel fail");

	cudaEventRecord(time_comp_stop, 0);
	cudaEventSynchronize(time_comp_stop);
	cudaEventElapsedTime(&time2, time_comp_start, time_comp_stop);
	cudaEventDestroy(time_comp_stop);
	cudaEventDestroy(time_comp_start);
	cout<<endl<<"*	2. Time for GPU computation: "<<time2<<"ms, GPU throughput: "<<packet_num/time2/1000<<" MPPS"<<endl;


	cudaEventRecord(time_memcpyD2H_start, 0);
	
	cudaMemcpy(bv_final, gpu_merge_results, sizeof(long int) * packet_num * int_count, cudaMemcpyDeviceToHost);
	
	cudaEventRecord(time_memcpyD2H_stop, 0);
	cudaEventSynchronize(time_memcpyD2H_stop);
	cudaEventElapsedTime(&time3, time_memcpyD2H_start, time_memcpyD2H_stop);
	cudaEventDestroy(time_memcpyD2H_stop);
	cudaEventDestroy(time_memcpyD2H_start);
	cout<<endl<<"*	3. Time for memcpy H2D: "<<time3<<"ms, Total bytes copied: "<<endl;
	cout<<"    		-> Bv_final: "<< sizeof(long int) * packet_num * int_count<<" Bytes"<<endl<<endl;

	cout<<endl<<">>>>>> Total GPU throughput: "<<packet_num/(time1 + time2 + time3)/1000<<" MPPS"<<endl;

	//data_test(tree, headers, bv, bv_final, packet_num, 8);

/********************************************************
*	Clear Memory:
*		1. Dynamic allocations on host
*		2. cudaFrees
********************************************************/
	cudaFree(gpu_tree);
	cudaCheckErrors("Free gpu_tree fail");
	cudaFree(gpu_bv);
	cudaCheckErrors("Free bv fail");
	cudaFree(gpu_headers);
	cudaCheckErrors("Free gpu_headers fail");
	cudaFree(gpu_match_result);
	cudaCheckErrors("Free gpu_match_result fail");
	cudaFree(gpu_merge_results);
	cudaCheckErrors("Free gpu_merge_results fail");

	for (int i = 0; i < FIELD; i++){
		delete tree[i];
	}
	for(int i = 0; i < FIELD; i++){
		delete headers[i];
	}
	for(int i = 0; i < FIELD*(RULE+1); i++){
		delete bv[i];
	}
	delete tree;
	delete bv;
	delete headers;
	delete bv_final;
	delete match_result;
	delete tree_flatten;
	delete headers_flatten;
	delete bv_flatten;
	delete merge_results;
    cout<<"============================ Experiment Ends ============================"<<endl;
	return 0;
}




void tree_gen(int** tree, int field, int rule){
	for(int i = 0; i < field; i++){
		tree[i][0] = rand() % 100;
		int temp[rule];
		temp[0] = tree[i][0];
		for (int j = 1; j < rule; j++){
			temp[j] = temp[j-1] + rand() % 20 + 1;
		}
		int temp_index = rule-1, tree_index = rule -1, level = log(rule+1) / log(2);
		int step_index = level;
		while (step_index >= 1){
			int step = pow(2, (level - step_index + 1));
			while (temp_index >= 0){
				tree[i][tree_index] = temp[temp_index];
				temp_index -= step;
				tree_index--;
			}
			step_index--;
			temp_index = rule - 1 - (pow(2, level - step_index) - 1);
		}
	}
}
void header_gen(int** headers, int** tree, int field, int packet_num){
	for (int i = 0; i < field; i++){
		for(int j = 0; j < packet_num; j++){
			headers[i][j] = rand() % 6000;
		}
	
	}
}
void bv_gen(long int ** bv, long int* bv_final, int packet_num){
	for (int i = 0; i < ALLRULE / sizeof(long int); i++){
		for (int j = 0; j < FIELD*(RULE+1); j++){
			bv[j][i] = rand() % 1000000;
		}
	}
	for(int i = 0; i < packet_num; i++){
		bv_final[i] = -1;
	}
}
void data_test(int** tree, int** headers, long int** bv, int* bv_final, int packet_num, int type){
	if (type > 15 | type == 0){
		return;
	}
	if (type % 2 == 1){
		cout<<"Tree: "<<endl;
		for(int i = 0; i < RULE; i++){
			cout<<"Line: "<<i<<": ";
			for(int j = 0; j < FIELD; j++){
				cout<<tree[j][i]<<" ";
			}
			cout<<endl;
		}
	}
	if (type % 4 == 2 | type % 4 == 3){
		cout<<endl<<"Headers: "<<endl;
		for(int i = 0; i < packet_num; i++){
			cout<<"Header "<<i<<": ";
			for(int j = 0; j < FIELD; j++){
				cout<<headers[j][i]<<" ";
			}
			cout<<endl;
		}
	}
	if (type % 8 == 4 | type % 8 == 5 | type % 8 == 6 | type % 8 == 7){
		cout<<endl<<"bv: "<<endl;
		for(int i = 0; i < ALLRULE; i++){
			cout<<"Line "<<i<<": ";
			for (int j = 0; j < FIELD*(RULE+1); j++){
			cout<<bv[j][i]<<" ";
			}
			cout<<endl;
		}
	}

	if (type > 7){
		cout<<endl<<"bv_final: "<<endl;
		for(int i = 0; i < packet_num; i++){
			cout<<bv_final[i]<<" ";
		}
		cout<<endl;
	}
	cout<<"============== End of Print =============="<<endl;
}



