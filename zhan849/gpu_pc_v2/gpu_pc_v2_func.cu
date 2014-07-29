#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cublas.h>
#include "gpu_pc_v2_func.h"

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
	cout<<"... Tree Gen ..."<<endl;
}

void header_gen(int** headers, int** tree, int field, int packet_num){
	for (int i = 0; i < field; i++){
		for(int j = 0; j < packet_num; j++){
			headers[i][j] = rand() % 6000;
		}
	
	}
	cout<<"... Header Gen ..."<<endl;
}

void bv_gen(long int** bv, long int* bv_final, int packet_num){
	for (int i = 0; i < int_count; i++){
		for (int j = 0; j < FIELD*(RULE+1); j++){
			bv[j][i] = rand() % 1000000;
		}
	}
	for(int i = 0; i < packet_num; i++){
		bv_final[i] = -1;
	}
	cout<<"... BV Gen ..."<<endl;
}
void bv_gen_short(int* bv, int* bv_final, int packet_num){
	for (int i = 0; i < FIELD*(RULE + 1)*int_count; i++){
		bv[i] = rand() % 5;
	}
	for(int i = 0; i < packet_num; i++){
		bv_final[i] = 1;
	}
	cout<<"... BV_Short Gen ..."<<endl;
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

__global__ void packet_classify(int* gpu_tree, int* gpu_headers, int* gpu_match_result, int packet_num){
	__shared__ int gpu_tree_shared[FIELD*RULE];
	int level = 0;
	while(level * block_dim + threadIdx.x < FIELD * RULE){
		gpu_tree_shared[level * block_dim + threadIdx.x] = gpu_tree[level * block_dim + threadIdx.x];
		level++;
	}
	__syncthreads();
	int i = 0;
	while (i < RULE){
		
		i = 2 * i + (gpu_headers[blockDim.x * blockIdx.x + threadIdx.x] <= gpu_tree_shared[(blockDim.x * blockIdx.x + threadIdx.x) / packet_num * RULE+i]) * 1 + (gpu_headers[blockDim.x * blockIdx.x + threadIdx.x] > gpu_tree_shared[(blockDim.x * blockIdx.x + threadIdx.x) / packet_num * RULE+i]) * 2;
	}
	gpu_match_result[blockDim.x * blockIdx.x + threadIdx.x] = i - RULE;

}
__global__ void pc_short(int* gpu_tree, int* gpu_headers, int* gpu_bv, int* gpu_bv_final, int packet_num){
	__shared__ int gpu_tree_shared[FIELD*RULE];
	__shared__ int gpu_bv_shared[FIELD*(RULE+1)*int_count];
	if (threadIdx.x < FIELD * RULE){
		gpu_tree_shared[threadIdx.x] = gpu_tree[threadIdx.x];
	}
	if (threadIdx.x >= FIELD * RULE && threadIdx.x <= FIELD * (RULE + 1) * int_count){
		gpu_bv_shared[threadIdx.x - FIELD * RULE] = gpu_bv[threadIdx.x - FIELD * RULE];
	}
	__syncthreads();
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	__shared__ int partial_result;
	partial_result = 0xffffffff;
	for (int j = 0; j < FIELD; j++){
		int i = 0;
		while (i < RULE){
			i = 2 * i + (gpu_headers[index * FIELD + j] <= gpu_tree_shared[index % FIELD * RULE + i]) * 1 + (gpu_headers[index * FIELD + j] > gpu_tree_shared[index % FIELD * RULE + i]) * 2;
		}
		partial_result &= gpu_bv_shared[i - RULE];
	}
	gpu_bv_final[ index ] = partial_result;
}
__global__ void packet_merge(long int* gpu_bv, int* gpu_match_result, long int* gpu_merge_result, long int*gpu_bv_final, int packet_num){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
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

}

void merge(void* foo){
	pthread_param_C* param = (pthread_param_C*) foo;
	for (int i = 0; i < param->BATCH; i++){
		//cout<<"[ Merge ] Thread: "<<param->thread_id<<", header # "<<i<<endl;
		for (int j = 0; j < int_count; j++){
			/*long int merge_partial = 0xffffffffffffffff;
			for (int k = 0; k < FIELD; k++){
				merge_partial &= param->merge_source[param->match_result[(param->thread_id * param->BATCH + i) * FIELD + k]][j];
				if (merge_partial == 0){
					break;
				}
			}
			if (merge_partial != 0){
				param->merge_result[(param->thread_id * param->BATCH + i) * int_count + j] = merge_partial;
				break;
			}*/
			param->merge_result[(param->thread_id * param->BATCH + i) * int_count + j] = param->merge_source[param->match_result[(param->thread_id * param->BATCH + i) * FIELD + 0]][j] & 
																						 param->merge_source[param->match_result[(param->thread_id * param->BATCH + i) * FIELD + 1]][j] &
																						 param->merge_source[param->match_result[(param->thread_id * param->BATCH + i) * FIELD + 2]][j] & 
																						 param->merge_source[param->match_result[(param->thread_id * param->BATCH + i) * FIELD + 3]][j] & 
																						 param->merge_source[param->match_result[(param->thread_id * param->BATCH + i) * FIELD + 4]][j] & 
																						 param->merge_source[param->match_result[(param->thread_id * param->BATCH + i) * FIELD + 5]][j];/*
																						 param->merge_source[param->match_result[(param->thread_id * param->BATCH + i) * FIELD + 6]][j] & 
																						 param->merge_source[param->match_result[(param->thread_id * param->BATCH + i) * FIELD + 7]][j] &
																						 param->merge_source[param->match_result[(param->thread_id * param->BATCH + i) * FIELD + 8]][j] & 
																						 param->merge_source[param->match_result[(param->thread_id * param->BATCH + i) * FIELD + 9]][j] & 
																						 param->merge_source[param->match_result[(param->thread_id * param->BATCH + i) * FIELD + 10]][j] & 
																						 param->merge_source[param->match_result[(param->thread_id * param->BATCH + i) * FIELD + 11]][j] &
																						 param->merge_source[param->match_result[(param->thread_id * param->BATCH + i) * FIELD + 12]][j] & 
																						 param->merge_source[param->match_result[(param->thread_id * param->BATCH + i) * FIELD + 13]][j] &
																						 param->merge_source[param->match_result[(param->thread_id * param->BATCH + i) * FIELD + 14]][j];*/
		}
	}
	//cout<<"Thread "<<param->thread_id<<" finish!"<<endl;
}

void partial_merge(void* foo){
	pthread_param_P* param = (pthread_param_P*) foo;

}

void final_merge(void* foo){
	pthread_param_F* param = (pthread_param_F*) foo;

}









