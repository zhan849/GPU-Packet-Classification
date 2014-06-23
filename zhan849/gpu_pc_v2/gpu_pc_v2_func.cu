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

__global__ void packet_classify(int* gpu_tree, int* gpu_headers, int* gpu_match_result, int packet_num, int block_dim){
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

__global__ void packet_merge(long int* gpu_bv, int* gpu_match_result, long int* gpu_merge_result, long int*gpu_bv_final, int packet_num, int block_dim){
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
	for (int i = 0; i < int_count; i++){
		param->merge_result[i] = param->partial_merge_source[0][i] & 
								 param->partial_merge_source[1][i] & 
								 param->partial_merge_source[2][i] & 
								 param->partial_merge_source[3][i] & 
								 param->partial_merge_source[4][i] & 
								 param->partial_merge_source[5][i] & 
								 param->partial_merge_source[6][i] & 
								 param->partial_merge_source[7][i] & 
								 param->partial_merge_source[8][i] & 
								 param->partial_merge_source[9][i] & 
								 param->partial_merge_source[10][i] & 
								 param->partial_merge_source[11][i] &
								 param->partial_merge_source[12][i] & 
								 param->partial_merge_source[13][i] & 
								 param->partial_merge_source[14][i]; 
	}
/*	param->bv_final[param->thread_id] = param->merge_result[0] &
										param->merge_result[1] &
										param->merge_result[2] &
										param->merge_result[3] &
										param->merge_result[4] &
										param->merge_result[5] &
										param->merge_result[6] &
										param->merge_result[7] &
										param->merge_result[8] &
										param->merge_result[9] &
										param->merge_result[10] &
										param->merge_result[11] &
										param->merge_result[12] &
										param->merge_result[13] &
										param->merge_result[14] &
										param->merge_result[15] &
										param->merge_result[16] &
										param->merge_result[17] &
										param->merge_result[18] &
										param->merge_result[19] &
										param->merge_result[20] &
										param->merge_result[21] &
										param->merge_result[22] &
										param->merge_result[23] &
										param->merge_result[24] &
										param->merge_result[25] &
										param->merge_result[26] &
										param->merge_result[27] &
										param->merge_result[28] &
										param->merge_result[29] &
										param->merge_result[30] &
										param->merge_result[31];
										*/
}

void partial_merge(void* foo){
	pthread_param_P* param = (pthread_param_P*) foo;
	param -> merge_result_partial[param -> thread_id] = param->merge_source[0] &
													   	param->merge_source[1] & 
													   	param->merge_source[2] & 
													   	param->merge_source[3] & 
													   	param->merge_source[4] & 
													   	param->merge_source[5] & 
													   	param->merge_source[6] & 
													   	param->merge_source[7] & 
													   	param->merge_source[8] & 
													   	param->merge_source[9] & 
													   	param->merge_source[10] & 
													   	param->merge_source[11] &
													   	param->merge_source[12] &
													   	param->merge_source[13] & 
													   	param->merge_source[14];   
}

void final_merge(void* foo){
	pthread_param_F* param = (pthread_param_F*) foo;
	param->bv_final[param->thread_id] = param->merge_source[0] &
										param->merge_source[1] &
										param->merge_source[2] &
										param->merge_source[3] &
										param->merge_source[4] &
										param->merge_source[5] &
										param->merge_source[6] &
										param->merge_source[7] &
										param->merge_source[8] &
										param->merge_source[9] &
										param->merge_source[10] &
										param->merge_source[11] &
										param->merge_source[12] &
										param->merge_source[13] &
										param->merge_source[14] &
										param->merge_source[15] &
										param->merge_source[16] &
										param->merge_source[17] &
										param->merge_source[18] &
										param->merge_source[19] &
										param->merge_source[20] &
										param->merge_source[21] &
										param->merge_source[22] &
										param->merge_source[23] &
										param->merge_source[24] &
										param->merge_source[25] &
										param->merge_source[26] &
										param->merge_source[27] &
										param->merge_source[28] &
										param->merge_source[29] &
										param->merge_source[30] &
										param->merge_source[31];
}









