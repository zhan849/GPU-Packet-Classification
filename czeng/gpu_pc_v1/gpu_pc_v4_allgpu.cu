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

#define FIELD 15
#define RULE 511
#define ALLRULE 2048
#define WSIZE 32
#define int_count ALLRULE / (sizeof(long int) * 8)

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

__global__ void packet_classify(int* gpu_tree, int* gpu_headers, int* gpu_match_result, int packet_num, int block_dim, long int* gpu_bv, long int* gpu_merge_result, long int*gpu_bv_final){
	__shared__ int gpu_tree_shared[FIELD*RULE];
	int level = 0;
	while(level * block_dim + threadIdx.x < FIELD * RULE){
		gpu_tree_shared[level * block_dim + threadIdx.x] = gpu_tree[level * block_dim + threadIdx.x];
		level++;
	}
	__syncthreads();
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int i = 0;
	if (index < packet_num * FIELD){
		// printf ("index = %d, packet_num %d\n", index, packet_num);
		i = 0;
		while (i < RULE){
			
			i = 2 * i + (gpu_headers[blockDim.x * blockIdx.x + threadIdx.x] <= gpu_tree_shared[(blockDim.x * blockIdx.x + threadIdx.x) / packet_num * RULE+i]) * 1 + (gpu_headers[blockDim.x * blockIdx.x + threadIdx.x] > gpu_tree_shared[(blockDim.x * blockIdx.x + threadIdx.x) / packet_num * RULE+i]) * 2;
		}
		gpu_match_result[blockDim.x * blockIdx.x + threadIdx.x] = i - RULE;
		// printf ("i = %d\n", i);
	}
	__syncthreads();

	// printf("The packet number is: %d, and the packet value is %d. %d Here is a float: %f\n", i, packet[i],intarray[i], floatarray[i]);
// };
// __global__ void packet_merge(long int* gpu_bv, int* gpu_match_result, long int* gpu_merge_result, long int*gpu_bv_final, int packet_num, int block_dim){
// // __global__ void packet_merge(long int* gpu_bv, long int* gpu_merge_result, long int*gpu_bv_final){
	int packetIdx = index/int_count;
	gpu_merge_result[index] = gpu_bv[gpu_match_result[packetIdx*15]*int_count + index%int_count] &
							  gpu_bv[gpu_match_result[packetIdx*15+1]*int_count + index%int_count] &
							  gpu_bv[gpu_match_result[packetIdx*15+2]*int_count + index%int_count] &
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

	__syncthreads();

	if (blockDim.x * blockIdx.x + threadIdx.x < packet_num){
		gpu_bv_final[blockDim.x*blockIdx.x+threadIdx.x] = gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)*int_count] & 
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)*int_count+1] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+2] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+3] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+4] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+5] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+6] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+7] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+8] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)*int_count+9] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+10] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+11] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+12] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+13] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+14] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+15] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+16] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)*int_count+17] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+18] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+19] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+20] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+21] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+22] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+23] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+24] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)*int_count+25] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+26] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+27] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+28] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+29] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+30] &
														  gpu_merge_result[(blockDim.x*blockIdx.x+threadIdx.x)%int_count+31];

	}
	
	

};

int main(int argc, char** argv){
	if(argc!=4){
		// cout<<"usage ./openflow  *Packet_num   *Grid_dim   *Block_dim   *Grid_dim_merge   *Block_dim_merge"<<endl; 
		
		cout<<"usage ./openflow  *Packet_num  *Grid_dim_merge   *Block_dim_merge"<<endl; 
		return 0;
	}
	int packet_num = atoi(argv[1]);
	// int grid_dim = atoi(argv[2]);
	// int block_dim = atoi(argv[3]);
	int grid_dim_merge = atoi(argv[2]);
	int block_dim_merge = atoi (argv[3]);
	// if (grid_dim*block_dim != packet_num*FIELD){
	// 	cout<<"ERROR: Total number of threads in stage 1 must equal packet_num * FIELD"<<endl;
	// 	return 1;
	// }
	// if (argv[2] * argv[3] != packet_num * int_count){
	// 	cout<<"ERROR: Total number of threads in stage 2 must equal packet_num * int_count"<<endl;
	// 	return 1;
	// }
cout<<"============================ Experiment Starts ============================"<<endl;
//	cout<<"grid_dim: "<<grid_dim<<", block_dim: "<<block_dim<<", packet_num: "<<packet_num;
//	cout<<", grid_dim_merge: "<<grid_dim_merge<<", block_dim_merge: "<<block_dim_merge<<endl;

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
			bv[i] = new long int[int_count];
		}
	long int* bv_final = new long int[packet_num];
	int* match_result = new int[packet_num * FIELD];
	long int* merge_result = new long int[int_count*packet_num];

	tree_gen(tree, FIELD, RULE);
	header_gen(headers, tree, FIELD, packet_num);
	bv_gen(bv, bv_final, packet_num);
	//data_test(tree, headers, bv, bv_final, packet_num, 3);

/********************************************************
*	Flatten All the 2D Arrays
********************************************************/
	int* tree_flatten = new int[RULE*FIELD];
	int* headers_flatten = new int[packet_num*FIELD];
	long int* bv_flatten = new long int[FIELD*(RULE+1) * int_count];

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
		for (int j = 0; j < int_count; j++){
			bv_flatten[ i * int_count + j] = bv[i][j];
		}
	}
/********************************************************
*	Declare cuda events for statistical purposes [Search]:
*		1. time_memcpyH2D
*		2. time_memcpyD2H
*		3. time_pc
********************************************************/
	float time1, time2, time3, time4;
	cudaEvent_t time_search_memcpyH2D_start, time_search_memcpyH2D_stop, time_merge_memcpyD2H_start, time_merge_memcpyD2H_stop, time_comp_start, time_comp_stop;
	cudaEventCreate(&time_search_memcpyH2D_start);
	cudaEventCreate(&time_search_memcpyH2D_stop);
	cudaEventCreate(&time_merge_memcpyD2H_start);
	cudaEventCreate(&time_merge_memcpyD2H_stop);
	cudaEventCreate(&time_comp_start);
	cudaEventCreate(&time_comp_stop);


/********************************************************
*	Allocate Space in Device:
*		1. gpu_tree 
*		2. gpu_bv 
*		3. gpu_bv_final
*		4. gpu_headers
********************************************************/
	dim3 dimGrid_merge(grid_dim_merge,1);
	dim3 dimBlock_merge(block_dim_merge,1);

	int* gpu_tree;
	int* gpu_headers;
	int* gpu_match_result;
	long int* gpu_bv_final;
	long int* gpu_merge_result;
	long int* gpu_bv;

// gpu_tree, gpu_headers, gpu_match_result, packet_num, block_dim_merge, gpu_bv, gpu_merge_result, gpu_bv_final
	cudaMalloc((void**)&gpu_tree, sizeof(int)*FIELD*RULE);
		cudaCheckErrors("cudaMalloc gpu_tree");
	cudaMalloc((void**)&gpu_headers, sizeof(int)*FIELD*packet_num);
		cudaCheckErrors("cudaMalloc gpu_headers");
	cudaMalloc((void**)&gpu_bv, sizeof(long int)*(RULE+1)*FIELD*int_count);
		cudaCheckErrors("cudaMalloc gpu_bv");
	cudaMalloc((void**)&gpu_match_result, sizeof(int)*packet_num*FIELD);
		cudaCheckErrors("cudaMalloc gpu_match_result");
	cudaMalloc((void**)&gpu_merge_result, sizeof(long int)*packet_num*int_count);
		cudaCheckErrors("cudaMalloc gpu_merge_result");
	cudaMalloc((void**)&gpu_bv_final, sizeof(long int)*packet_num);
		cudaCheckErrors("cudaMalloc gpu_bv_final");

	cudaEventRecord(time_search_memcpyH2D_start, 0);
	
	cudaMemcpy(gpu_tree, tree_flatten, sizeof(int)*RULE*FIELD, cudaMemcpyHostToDevice);
		cudaCheckErrors("cudaMemcpy gpu_tree");
	cudaMemcpy(gpu_headers, headers_flatten, sizeof(int)*FIELD*packet_num, cudaMemcpyHostToDevice);
		cudaCheckErrors("cudaMemcpy gpu_headers");
	cudaMemcpy(gpu_bv, bv_flatten, sizeof(long int)*(RULE+1)*FIELD*int_count, cudaMemcpyHostToDevice);
		cudaCheckErrors("cudaMemcpy merge gpu_bv");

	cudaEventRecord(time_search_memcpyH2D_stop, 0);
	cudaEventSynchronize(time_search_memcpyH2D_stop);
	cudaEventElapsedTime(&time1, time_search_memcpyH2D_start, time_search_memcpyH2D_stop);
	cudaEventDestroy(time_search_memcpyH2D_stop);
	cudaEventDestroy(time_search_memcpyH2D_start);
	cout<<endl<<"Number of Packets: "<<packet_num<<endl;
	cout<<endl<<">>>>>>[Stage 1: Search] "<<"grid_dim: "<<grid_dim_merge<<", block_dim: "<<block_dim_merge<<endl;
	cout<<endl<<"*	1. Time for memcpy H2D: "<<time1<<"ms, Total bytes copied: "<<endl;
	cout<<"    		-> Tree: "<< sizeof(int)*RULE*FIELD<<" Bytes"<<endl;
	cout<<"    		-> Headers: "<< sizeof(int)*FIELD*packet_num<<" Bytes"<<endl;
	cout<<"    		-> BV: "<< sizeof(long int)*(RULE+1)*FIELD*int_count <<" Bytes"<<endl;
	cout<<"    		-> Total Memory Copy: "<< sizeof(int)*RULE*FIELD + sizeof(int)*FIELD*packet_num + sizeof(long int)*(RULE+1)*FIELD*int_count <<" Bytes"<<endl;



/********************************************************
*	Main Packet Classification Process [Search]
*		1. Function Call
*		2. Timing
*		3. Memory copy back (gpu_bv_final)
********************************************************/
/********************************************************
*	Main Packet Classification Process [Merge]
*		1. Function Call
*		2. Timing
*		3. Memory copy back (gpu_bv_final)
// ********************************************************/
	cudaEventRecord(time_comp_start, 0);
	packet_classify<<<dimGrid_merge, dimBlock_merge>>>(gpu_tree, gpu_headers, gpu_match_result, packet_num, block_dim_merge, gpu_bv, gpu_merge_result, gpu_bv_final);
	cudaCheckErrors("Computation fail");

	cudaEventRecord(time_comp_stop, 0);
	cudaEventSynchronize(time_comp_stop);
	cudaEventElapsedTime(&time2, time_comp_start, time_comp_stop);
	cudaEventDestroy(time_comp_stop);
	cudaEventDestroy(time_comp_start);
	cout<<endl<<"*	2. Time for GPU computation: "<<time2<<"ms, GPU throughput: "<<packet_num/time2/1000<<" MPPS"<<endl;


	// cudaEventRecord(time_search_memcpyD2H_start, 0);
	
	// cudaMemcpy(match_result, gpu_match_result, sizeof(int)*packet_num*FIELD, cudaMemcpyDeviceToHost);

	
	// cudaEventRecord(time_search_memcpyD2H_stop, 0);
	// cudaEventSynchronize(time_search_memcpyD2H_stop);
	// cudaEventElapsedTime(&time3, time_search_memcpyD2H_start, time_search_memcpyD2H_stop);
	// cudaEventDestroy(time_search_memcpyD2H_stop);
	// cudaEventDestroy(time_search_memcpyD2H_start);
	// cout<<endl<<"*	3. Time for memcpy D2H: "<<time3<<"ms, Total bytes copied: "<<endl;
	// cout<<"    		-> Match_result: "<< sizeof(int)*packet_num*FIELD<<" Bytes"<<endl<<endl;

	//data_test(tree, headers, bv, bv_final, packet_num, 8);



/********************************************************
*	Declare cuda events for statistical purposes [Search]:
*		1. time_merge_memcpyH2D
*		2. time_merge_memcpyD2H
*		3. time_mg
********************************************************/

	// dim3 dimGrid_merge(grid_dim_merge,1);
	// dim3 dimBlock_merge(block_dim_merge,1);
	// float time4, time5, time6, time7;
	// cudaEvent_t time_merge_memcpyH2D_start, time_merge_memcpyH2D_stop, time_merge_memcpyD2H_start, time_merge_memcpyD2H_stop, time_merge_start, time_merge_stop;
	// cudaEvent_t time_clean_start, time_clean_stop;
	// cudaEventCreate(&time_merge_memcpyH2D_start);
	// cudaEventCreate(&time_merge_memcpyH2D_stop);
	// cudaEventCreate(&time_merge_memcpyD2H_start);
	// cudaEventCreate(&time_merge_memcpyD2H_stop);
	// cudaEventCreate(&time_merge_start);
	// cudaEventCreate(&time_merge_stop);
	// cudaEventCreate(&time_clean_start);
	// cudaEventCreate(&time_clean_stop);
	// cout<<"---------------------------------------------------------"<<endl;
	// cout<<endl<<">>>>>>[Stage 2: Merge]"<<" grid_dim_merge: "<<grid_dim_merge<<", block_dim_merge: "<<block_dim_merge<<endl;



	// cudaEventRecord(time_merge_memcpyH2D_start, 0);
	// cudaMemcpy(gpu_bv, bv_flatten, sizeof(long int)*(RULE+1)*FIELD*int_count, cudaMemcpyHostToDevice);
	// // 	cudaCheckErrors("cudaMemcpy merge gpu_bv");
	// cudaMemcpy(gpu_match_result, match_result, sizeof(int)*packet_num*FIELD, cudaMemcpyHostToDevice);
	// 	cudaCheckErrors("cudaMemcpy merge gpu_match_result");

	// cudaEventRecord(time_merge_memcpyH2D_stop, 0);
	// cudaEventSynchronize(time_merge_memcpyH2D_stop);
	// cudaEventElapsedTime(&time4, time_merge_memcpyH2D_start, time_merge_memcpyH2D_stop);
	// cudaEventDestroy(time_merge_memcpyH2D_stop);
	// cudaEventDestroy(time_merge_memcpyH2D_start);

	// cout<<endl<<"*	1. Time for memcpy H2D: "<<time4<<"ms, Total bytes copied: "<<endl;
	// cout<<"    		-> BV: "<< sizeof(long int)*(RULE+1)*FIELD*int_count <<" Bytes"<<endl;
	// cout<<"    		-> match_result: "<< sizeof(int)*FIELD*packet_num<<" Bytes"<<endl;
	// cout<<"    		-> Total Memory Copy: "<< sizeof(int)*FIELD*packet_num + sizeof(long int)*(RULE+1)*FIELD*int_count<<" Bytes"<<endl;
/********************************************************
*	Main Packet Classification Process [Merge]
*		1. Function Call
*		2. Timing
*		3. Memory copy back (gpu_bv_final)
// ********************************************************/
// 	cudaEventRecord(time_merge_start, 0);

	// packet_merge<<<dimGrid_merge, dimBlock_merge>>>(gpu_bv, gpu_match_result, gpu_merge_result, gpu_bv_final, packet_num, block_dim, gpu_merge_result);
	// cudaCheckErrors("Merge fail");

	// cudaEventRecord(time_merge_stop, 0);
	// cudaEventSynchronize(time_merge_stop);
	// cudaEventElapsedTime(&time5, time_merge_start, time_merge_stop);
	// cudaEventDestroy(time_merge_stop);
	// cudaEventDestroy(time_merge_start);
	// cout<<endl<<"*	2. Time for GPU computation: "<<time5<<"ms, GPU throughput: "<<packet_num/time5/1000<<" MPPS"<<endl;



	cudaEventRecord(time_merge_memcpyD2H_start, 0);
	
	cudaMemcpy(bv_final, gpu_bv_final, sizeof(long int)*packet_num, cudaMemcpyDeviceToHost);
		cudaCheckErrors("Cuda Memcpy D2H merge fail");
	
	cudaEventRecord(time_merge_memcpyD2H_stop, 0);
	cudaEventSynchronize(time_merge_memcpyD2H_stop);
	cudaEventElapsedTime(&time3, time_merge_memcpyD2H_start, time_merge_memcpyD2H_stop);
	cudaEventDestroy(time_merge_memcpyD2H_stop);
	cudaEventDestroy(time_merge_memcpyD2H_start);
	cout<<endl<<"*	3. Time for memcpy D2H: "<<time3<<"ms, Total bytes copied: "<<endl;
	cout<<"    		-> bv_final: "<< sizeof(long int)*packet_num<<" Bytes"<<endl<<endl;



/********************************************************
*	Clear Memory:
*		1. Dynamic allocations on host
*		2. cudaFrees
********************************************************/
	cudaEvent_t time_clean_start, time_clean_stop;
	cudaEventCreate(&time_clean_start);
	cudaEventCreate(&time_clean_stop);
	cudaEventRecord(time_clean_start, 0);

	cudaFree(gpu_tree);
	cudaCheckErrors("Free gpu_tree fail");
	cudaFree(gpu_headers);
	cudaCheckErrors("Free gpu_headers fail");
	cudaFree(gpu_bv);
	cudaCheckErrors("Free bv fail");
	cudaFree(gpu_bv_final);
	cudaCheckErrors("Free gpu_bv_final fail");
	cudaFree(gpu_match_result);
	cudaCheckErrors("Free gpu_match_result fail");
	cudaFree(gpu_merge_result);
	cudaCheckErrors("Free gpu_merge_result fail");

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
	delete merge_result;

	cudaEventRecord(time_clean_stop, 0);
	cudaEventSynchronize(time_clean_start);
	cudaEventElapsedTime(&time4, time_clean_start, time_clean_stop);
	cudaEventDestroy(time_clean_stop);
	cudaEventDestroy(time_clean_start);
	cout<<endl<<"*	4. Time for cleaning memory: "<<time4<<"ms."<<endl<<endl;



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
void bv_gen(long int** bv, long int* bv_final, int packet_num){
	for (int i = 0; i < int_count; i++){
		for (int j = 0; j < FIELD*(RULE+1); j++){
			bv[j][i] = rand() % 1000000;
		}
	}
//	for(int i = 0; i < packet_num; i++){
//		bv_final[i] = -1;
//	}
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



