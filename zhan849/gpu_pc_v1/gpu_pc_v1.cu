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
#define RULE 31
#define ALLRULE 31
#define WSIZE 32

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
void bv_gen(bool**, int*, int);
void data_test(int**, int**, bool**, int*, int);


int main(int argc, char** argv){
	if(argc!=4){
		cout<<"usage ./openflow  *Packet_num   *Grid_dim   *Block_dim"<<endl; 
		return 0;
	}
	int packet_num = atoi(argv[1]);
	int grid_dim = atoi(argv[2]);
	int block_dim = atoi(argv[3]);
	cout<<"grid_dim: "<<grid_dim<<", block_dim: "<<block_dim<<", packet_num: "<<packet_num<<endl;

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
	int** headers = new int*[packet_num];
		for (int i = 0; i < packet_num; i++){
			headers[i] = new int[FIELD];
		}
	bool** bv = new bool*[FIELD*(RULE+1)];
		for(int i = 0; i < FIELD*(RULE+1); i++){
			bv[i] = new bool[ALLRULE];
		}
	int* bv_final = new int[packet_num];
	printf("check point 1\n");
	tree_gen(tree, FIELD, RULE);
	printf("check point 2\n");
	header_gen(headers, tree, FIELD, packet_num);
	printf("check point 3\n");
	bv_gen(bv, bv_final, packet_num);
	printf("check point 4\n");
	data_test(tree, headers, bv, bv_final, packet_num);











/********************************************************
*	Clear Memory:
*		1. Dynamic allocations on host
*		2. cudaFrees
********************************************************/
	for (int i = 0; i < FIELD; i++){
		delete tree[i];
	}
	for(int i = 0; i < packet_num; i++){
		delete headers[i];
	}
	for(int i = 0; i < FIELD*(RULE+1); i++){
		delete bv[i];
	}
	delete tree;
	delete bv;
	delete headers;
	delete bv_final;

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
		cout<<"level: "<<level<<endl;
		cout<<"Tree #"<<i<<": "<<endl;
		while (step_index >= 1){
			int step = pow(2, (level - step_index + 1));
			while (temp_index >= 0){
				cout<<"temp_index: "<<temp_index<<", tree_index: "<<tree_index<<", step_index: "<<step_index<<", step: "<<step<<", value: "<<temp[temp_index]<<endl;
				tree[i][tree_index] = temp[temp_index];
				temp_index -= step;
				tree_index--;
			}
			step_index--;
			temp_index = rule - 1 - (pow(2, level - step_index) - 1);
			cout<<"New step_index = "<<step_index<<", New temp_index: "<<temp_index<<endl;
		}
		cout<<"end of tree#"<<i<<endl;
	}
}
void header_gen(int** headers, int** tree, int field, int packet_num){
	for (int i = 0; i < packet_num; i++){
		for(int j = 0; j < field; j++){
			headers[i][j] = rand() % 1000;
		}
		//printf("check point 3.%d\n", i);
	}
	for (int i = 0; i < field; i++){
		headers[0][i] = tree[i][12];
		headers[1][i] = tree[i][20];
		headers[2][i] = tree[i][100];
		headers[3][i] = tree[i][116];
		headers[4][i] = tree[i][234];
		headers[5][i] = tree[i][101];
		headers[6][i] = tree[i][28];
		headers[7][i] = tree[i][209];
		headers[8][i] = tree[i][131];
		headers[9][i] = tree[i][190];
		headers[10][i] = tree[i][57];
		headers[11][i] = tree[i][155];
		/*
		headers[173][i] = tree[i][312];
		headers[299][i] = tree[i][20];
		headers[287][i] = tree[i][100];
		headers[356][i] = tree[i][116];
		headers[390][i] = tree[i][234];
		headers[457][i] = tree[i][101];
		headers[460][i] = tree[i][188];
		headers[490][i] = tree[i][209];
		headers[555][i] = tree[i][131];
		headers[670][i] = tree[i][490];
		headers[799][i] = tree[i][257];
		headers[891][i] = tree[i][355];
		*/
	}
}
void bv_gen(bool** bv, int* bv_final, int packet_num){
	for (int i = 0; i < ALLRULE; i++){
		for (int j = 0; j < FIELD*(RULE+1); j++){
			int randnum = rand()%10;
			if(randnum == 1){
				bv[j][i] = true;
			}else{
				bv[j][i] = false;
			}
		}
	}
	for(int i = 0; i < packet_num; i++){
		bv_final[i] = -1;
	}
}
void data_test(int** tree, int** headers, bool** bv, int* bv_final, int packet_num){
	cout<<"Tree: "<<endl;
	cout<<tree[15][0]<<endl;
/*
	for(int i = 0; i < RULE; i++){
		cout<<"Line: "<<i<<": ";
		for(int j = 0; j < FIELD; j++){
			cout<<tree[i][j]<<" ";
		}
		cout<<endl;
	}
*/	
	cout<<endl<<"Headers: "<<endl;
	for(int i = 0; i < packet_num; i++){
		cout<<"Header "<<i<<": ";
		for(int j = 0; j < FIELD; j++){
			cout<<headers[i][j]<<" ";
		}
		cout<<endl;
	}
	cout<<endl<<"bv: "<<endl;
	for(int i = 0; i < ALLRULE; i++){
		cout<<"Line "<<i<<": ";
		for (int j = 0; j < FIELD*(RULE+1); j++){
		cout<<bv[j][i]<<" ";
		}
		cout<<endl;
	}
	cout<<endl<<"bv_final: "<<endl;
	for(int i = 0; i < packet_num; i++){
		cout<<bv_final[i]<<" ";
	}
	cout<<endl;
}



