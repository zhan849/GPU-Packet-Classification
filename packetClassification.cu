#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cublas.h>
//#define field 15
//#define rule 31

#define field 15
#define rule 511
using namespace std;

__host__ int** tree_gen(int f, int r){
	cout<<"field: "<<f<<", Rule: "<<r<<endl;
	srand(time(NULL));
	int** tree = new int*[f];
	for (int i = 0; i < f; i++){
		tree[i] = new int[r];
		for (int j = 0; j < r; j++){
			tree[i][j] = rand()%1000;
		}
	}
	return tree;
};
__host__ int** headers_gen(int num, int f){
	int** headers = new int*[f];
	for (int i = 0; i < f; i++){
		headers[i] = new int[num];
		for (int j = 0; j < num; j++){
			headers[i][j] = rand()%1000;
		}
	}
	return headers;
};
__global__ void packet_classify(int* gpu_tree, int* gpu_headers, int* gpu_bv, int* gpu_bv_final, int packet_num, int block_dim){
// For each block, copy the headers and bvs into the shared memory
	/*
	__shared__ int tree_shared[rule * field];
	
	int level = 0;
	while(level + threadIdx.x < rule * field){
		tree_shared[level + threadIdx.x] = gpu_tree[level + threadIdx.x];
		level = level + block_dim;
	}
	__syncthreads();
*/
	int i = 0, value, max_packet = block_dim/field; 
	
	//int packetID = (blockIdx.x * blockDim.x + threadIdx.x)/field;
	//int fieldID = (blockIdx.x * blockDim.x + threadIdx.x) % field;
	if (threadIdx.x / field < max_packet){
		//if (blockIdx.x * blockDim.x + threadIdx.x / field > packet_num){
		//	return;
		//}
		
		//i = 0;
		value = gpu_tree[(blockIdx.x * blockDim.x + threadIdx.x) % field * rule + i];
		//value = tree_shared[(blockIdx.x * blockDim.x + threadIdx.x) % field * rule + i];
		
		while(i < rule){
			
			if (gpu_headers[(blockIdx.x * blockDim.x + threadIdx.x) % field * packet_num + (blockIdx.x * blockDim.x + threadIdx.x)/field] == value){
				gpu_bv[(blockIdx.x * blockDim.x + threadIdx.x) % field * packet_num + (blockIdx.x * blockDim.x + threadIdx.x)/field] = i;
				i = rule;
			}
			else{
				if (gpu_headers[(blockIdx.x * blockDim.x + threadIdx.x) % field * packet_num + (blockIdx.x * blockDim.x + threadIdx.x)/field] < value){
					i = 2 * i + 1;
				}else {
					i = 2 * i + 2;
				}
				value = gpu_tree[(blockIdx.x * blockDim.x + threadIdx.x) % field * rule + i];
				//value = tree_shared[(blockIdx.x * blockDim.x + threadIdx.x) % field * rule + i];
			}

		}
	}
	

	__syncthreads();
/*
	for (int i = 0; i < field; i++){
		gpu_bv_final[i] = gpu_bv[i * packet_num + 28];
	}
	for (int i = field; i < 2*field; i++){
		gpu_bv_final[i] = gpu_bv[i%field * packet_num + 63];
	}
	for (int i = 2*field; i < 3*field; i++){
		gpu_bv_final[i] = gpu_bv[i%field * packet_num + 129];
	}

	//if(blockIdx.x * blockDim.x + threadIdx.x < packet_num){
	//	gpu_bv_final[blockIdx.x * blockDim.x + threadIdx.x] = gpu_bv[blockIdx.x * blockDim.x + threadIdx.x];
	//}
*/

	if (blockIdx.x * blockDim.x + threadIdx.x < packet_num){
		//int index = blockIdx.x * blockDim.x + threadIdx.x;
		gpu_bv_final[blockIdx.x * blockDim.x + threadIdx.x] = gpu_bv[blockIdx.x * blockDim.x + threadIdx.x];
		for (int i = 1; i < field; i++){
			gpu_bv_final[blockIdx.x * blockDim.x + threadIdx.x] = gpu_bv_final[blockIdx.x * blockDim.x + threadIdx.x] & gpu_bv[i * packet_num + blockIdx.x * blockDim.x + threadIdx.x]; 
		}
	}

/*
	for (int i = 0; i < packet_num; i++){
		gpu_bv_final[i] = gpu_bv[i];
	}

	for (int i = 1; i < field; i++){
		for (int j = 0; j < packet_num; j++){
			gpu_bv_final[j] = gpu_bv_final[j] & gpu_bv[i * packet_num + j];
		}
	}
*/
	//gpu_bv_final[packetID] = gpu_bv[packetID];
	//}		
		
		//}
		
	//}




};
int main(int argc, char** argv) {
	if(argc!=4){
		cout<<"usage ./openflow  *Packet_num   *Grid_dim   *Block_dim"<<endl; 
		return 0;
	}
	int packet_num = atoi(argv[1]);
	int grid_dim = atoi(argv[2]);
	int block_dim = atoi(argv[3]);
	cout<<"grid_dim: "<<grid_dim<<", block_dim: "<<block_dim<<", packet_num: "<<packet_num<<endl;





// Generating random packet headers and routing tables
	srand(time(NULL));
	int tree[field][rule];
//	int headers[field][packet_num];
	int** headers = new int*[field];
	int** bv = new int*[field];
	for (int i = 0; i < field; i++){
		headers[i] = new int[packet_num];
		bv[i] = new int[packet_num];
	}
	int* bv_final = new int[packet_num];
	//int bv[field][packet_num];
	//int bv_final[packet_num];
	cout<<"Check Point: allocate variables"<<endl;
	for (int i = 0; i < packet_num; i++){
		bv_final[i] = -1;
	}
	for(int i = 0; i < field; i++){
		tree[i][0] = rand() % 100;
		int temp[rule];
		temp[0] = tree[i][0];
		for (int j = 1; j < rule; j++){
			temp[j] = temp[j-1] + rand() % 20 + 1;
		}
		int temp_index = rule-1, tree_index = rule -1, level = log(rule+1) / log(2);
		int step_index = level;
		//cout<<"level: "<<level<<endl;
		//cout<<"Tree #"<<i<<": "<<endl;
		while (step_index >= 1){
			int step = pow(2, (level - step_index + 1));
			while (temp_index >= 0){
		//		cout<<"temp_index: "<<temp_index<<", tree_index: "<<tree_index<<", step_index: "<<step_index<<", step: "<<step<<endl;
				tree[i][tree_index] = temp[temp_index];
				temp_index -= step;
				tree_index--;
			}
			step_index--;
			temp_index = rule - 1 - (pow(2, level - step_index) - 1);
		//	cout<<"New step_index = "<<step_index<<", New temp_index: "<<temp_index<<endl;
		}
		//cout<<"end of tree#"<<i<<endl;
	}
	cout<<"Check point: Tree gen"<<endl;
	/*for (int i = 0; i < rule; i++){
		cout<<"Rule "<<i<<": ";
		for (int j = 0; j < field; j++){
			cout<<tree[j][i]<<" ";
		}
		cout<<endl;
	}*/
	for (int i = 0; i < packet_num; i++){
		for(int j = 0; j < field; j++){
			headers[j][i] = rand() % 500;
		}
	}
	cout<<"Check point: header gen"<<endl;
	// FOR TESTING, make some packets same to some rule sets
	for (int i = 0; i < field; i++){
		//headers[i][5] = tree[i][7];
		headers[i][0] = tree[i][12];
		headers[i][1] = tree[i][20];
		headers[i][2] = tree[i][100];
		headers[i][3] = tree[i][116];
		headers[i][4] = tree[i][234];
		headers[i][5] = tree[i][101];
		headers[i][6] = tree[i][28];
		headers[i][7] = tree[i][209];
		headers[i][8] = tree[i][131];
		headers[i][9] = tree[i][190];
		headers[i][10] = tree[i][457];
		headers[i][11] = tree[i][155];
		headers[i][173] = tree[i][312];
		headers[i][299] = tree[i][20];
		headers[i][287] = tree[i][100];
		headers[i][356] = tree[i][116];
		headers[i][390] = tree[i][234];
		headers[i][457] = tree[i][101];
		headers[i][460] = tree[i][188];
		headers[i][490] = tree[i][209];
		headers[i][555] = tree[i][131];
		headers[i][670] = tree[i][490];
		headers[i][799] = tree[i][257];
		headers[i][891] = tree[i][355];
	}
	cout<<"check point: set test headers"<<endl;
// Test packet headers and routing tables 
/*
	cout<<"Tree: "<<endl;
	for (int i = 0; i < rule; i++){
		cout<<"Rule "<<i<<": ";
		for (int j = 0; j < field; j++){
			cout<<tree[j][i]<<" ";
		}
		cout<<endl;
	}


	cout<<endl<<"Headers: "<<endl;
	for (int i = 0; i < packet_num; i++){
		for (int j = 0; j < field; j++){
			cout<<headers[j][i]<<" ";
		}
		cout<<endl;
	}
*/
	cout<<"End of Prints"<<endl;

// Define classification variables
	dim3 dimGrid(grid_dim,1);
	dim3 dimBlock(block_dim,1);
	int* gpu_tree;
	int* gpu_headers;
	int* gpu_bv;
	int* gpu_bv_final;
// Define timer variables
	float time1, time2;
	cudaEvent_t start2, stop2, start1, stop1;

cout<<"Check point 1"<<endl;
// Allocate memories for headers and trees
	// Trees
cudaMalloc((void**)&gpu_tree, sizeof(int)*size_t(rule*field));
cudaMalloc((void**)&gpu_headers, sizeof(int)*size_t(packet_num*field));
cudaMalloc((void**)&gpu_bv, sizeof(int)*size_t(packet_num*field));

cout<<"Check point 2"<<endl;
	// Fields

	cudaMalloc((void**)&gpu_bv_final, sizeof(int)*packet_num);

// Timers
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
			
			

cout<<"Check point 3"<<endl;
// Copy data to GPU global memory

    cudaEventRecord(start1, 0);
    cudaMemcpy(gpu_tree, tree, sizeof(int)*size_t(rule*field), cudaMemcpyHostToDevice);
    cout<<"Check point 4"<<endl;
    cudaMemcpy(gpu_headers, headers, sizeof(int)*size_t(packet_num*field), cudaMemcpyHostToDevice);
    cout<<"Check point 5"<<endl;
    cudaMemcpy(gpu_bv, bv, sizeof(int)*size_t(packet_num*field), cudaMemcpyHostToDevice);
    cout<<"Check point 6"<<endl;
	cudaMemcpy(gpu_bv_final, bv_final, sizeof(int)*packet_num, cudaMemcpyHostToDevice);

	cudaEventRecord(start2,0);
	cout<<"Check point 7"<<endl;
	packet_classify<<<dimGrid, dimBlock>>>(gpu_tree, gpu_headers, gpu_bv, gpu_bv_final, packet_num, block_dim);

	
	cudaEventRecord(stop2,0);
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&time2,start2,stop2);
	cudaEventDestroy(start2);
	cudaEventDestroy(stop2);	
			cout<<"Grid Size: "<<grid_dim<<"*1; Block Size: "<<block_dim<<"*1; Time for computation: "<<time2<<"ms; ";

	cudaMemcpy(bv_final, gpu_bv_final, sizeof(int)*packet_num, cudaMemcpyDeviceToHost);


	cudaEventRecord(stop1,0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&time1,start1,stop1);
	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);

	cout<<"Total Time: "<<time1<<"ms. "<<"Throughput: "<<packet_num/time2/1000<<" million packets / second."<<endl;
/*
	for(int i = 0; i < packet_num; i++){
		cout<<bv_final[i]<<" ";
	}
	cout<<endl;
*/
// Memory Deallocation
	cudaFree(gpu_tree);
	cudaFree(gpu_bv);
	cudaFree(gpu_headers);
	cudaFree(gpu_bv_final);
	for (int i = 0; i < field; i++){
	//	delete tree[i];
	//	delete headers[i];
		delete bv[i];
	} 
	//delete tree;
	//delete headers;
	delete bv;
	delete bv_final;

	return 0;
}