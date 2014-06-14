#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cublas.h>
using namespace std;
__global__ void packet_classify(int *gpu_tree1, int *gpu_tree2, int threadsize, unsigned long long int *gpu_bv1, unsigned long long int *gpu_bv2, unsigned long long int *gpu_final_bv, int *gpu_field1, int *gpu_field2)
{
	int value, i, j;



	unsigned long long int local_bv;
	__shared__ int tree1_s[15];
	__shared__ int tree2_s[15];
	__shared__ unsigned long long int bv1_s[15];
	__shared__ unsigned long long int bv2_s[15];
	
	
	if (threadIdx.x < 15){
		tree1_s[threadIdx.x] = gpu_tree1[threadIdx.x];
		//__syncthreads();
		tree2_s[threadIdx.x] = gpu_tree2[threadIdx.x];
		//__syncthreads();
		bv1_s[threadIdx.x] = gpu_bv1[threadIdx.x];
		//__syncthreads();
		bv2_s[threadIdx.x] = gpu_bv2[threadIdx.x];
	}
		__syncthreads();
	
	
	j = 0;
	i = 0;
	for(j=0;j<threadsize;j+=1)
	{
		//if((threadsize*(blockDim.x*blockIdx.x＋threadIdx.x) + j)>4095)
		//return;
		if (threadsize * (blockDim.x * blockIdx.x + threadIdx.x) + j > 65535){
			return;
		}
		local_bv = 0;
		value = tree1_s[i];
		while(i<15)
		{
			if(gpu_field1[threadsize * (blockDim.x * blockIdx.x + threadIdx.x) + j]== tree1_s[i])
				{
				local_bv = bv1_s[i];
				i = 15;
				}
			else
				{
				i = 2*i + ((gpu_field1[threadsize * (blockDim.x * blockIdx.x + threadIdx.x) + j]<value)? 1 : 2);
				value = tree1_s[i];
				}
		}
		
		i=0;
		value = tree2_s[i];
		while(i<15)
			{
			if(gpu_field2[threadsize * (blockDim.x * blockIdx.x + threadIdx.x) + j]== tree2_s[i])
				{
				local_bv = local_bv & bv2_s[i];
				i = 15;
				}
			else
				{
				i = 2*i + ((gpu_field2[threadsize * (blockDim.x * blockIdx.x + threadIdx.x) + j]<value)? 1 : 2);
				value = tree2_s[i];
				}
			}
		gpu_final_bv[threadsize * (blockDim.x * blockIdx.x + threadIdx.x) + j] = local_bv;
		//__syncthreads();
		i = 0;
	}	
	

}



unsigned long long int power(unsigned long long int a, unsigned long long int b){
	unsigned long long int result = 1;
	int i;
	for(i=0; i<b; i++)
		result= result*a;
	return result;
}	

int main(int argc, char ** argv){
	if(argc!=3){
		cout<<"usage ./openflow   *Grid_size   *Block_size"<<endl; 
		return 0;
	}
	int grid_dim = atoi(argv[1]);
	int block_dim = atoi(argv[2]);
		//for (int t = 16; t <= 256; t = t+16){
		FILE *fp;
		int i, j, k;
		int Tree1[15], Tree2[15];
		unsigned long long int BV1[15], BV2[15], final_bv[65536];
		unsigned int field1[65536], field2[65536]; // used to store the packet field
		
		
		
		Tree1[0]=1462; 	BV1[0]= power(2,0)+power(2,1); 									Tree2[0]=80;    BV2[0]= power(2,16)+power(2,21);
		Tree1[1]=967; 	BV1[1]= power(2,2)+power(2,6);									Tree2[1]=41;    BV2[1]= power(2,17)+power(2,28)+power(2,4);
		Tree1[2]=1563; 	BV1[2]= power(2,16)+power(2,21);								Tree2[2]=121;   BV2[2]= power(2,13);
		Tree1[3]=387; 	BV1[3]= power(2,15)+power(2,31);								Tree2[3]=23;    BV2[3]= power(2,19)+power(2,12);
		Tree1[4]=1042; 	BV1[4]= power(2,10)+power(2,18)+power(2,22);		Tree2[4]=52;    BV2[4]= power(2,27); 
		Tree1[5]=1477; 	BV1[5]= power(2,7)+power(2,8)+power(2,9);				Tree2[5]=94;    BV2[5]= power(2,3)+power(2,18)+power(2,22);	; 
		Tree1[6]=1870; 	BV1[6]= power(2,17)+power(2,28)+power(2,4);			Tree2[6]=130;   BV2[6]= power(2,10);
		Tree1[7]=12; 		BV1[7]= power(2,3);															Tree2[7]=12;    BV2[7]= power(2,7)+power(2,8)+power(2,9);
		Tree1[8]=497; 	BV1[8]= power(2,11)+power(2,5)+power(2,14);			Tree2[8]=25;    BV2[8]= power(2,11)+power(2,14);
		Tree1[9]=1011; 	BV1[9]= power(2,12);														Tree2[9]=45;    BV2[9]= power(2,15)+power(2,20); 
		Tree1[10]=1300; BV1[10]= power(2,13);														Tree2[10]=55;   BV2[10]= power(2,22)+power(2,23)+power(2,24);
		Tree1[11]=1465; BV1[11]= power(2,26)+power(2,25)+power(2,24);		Tree2[11]=85;   BV2[11]= power(2,31)+power(2,26);
		Tree1[12]=1500; BV1[12]= power(2,29);														Tree2[12]=95;   BV2[12]= power(2,30)+power(2,25);
		Tree1[13]=1600; BV1[13]= power(2,30);														Tree2[13]=126;  BV2[13]= power(2,2)+power(2,29);
		Tree1[14]=1983; BV1[14]= power(2,19)+power(2,27);								Tree2[14]=172;  BV2[14]= power(2,0)+power(2,1)+power(2,5)+power(2,6);
		//printf("%d\n", BV2[14]);

		// read packet data
		if ((fp=fopen("packet.txt","r"))==NULL)
			printf("Cannot open file. Check the name.\n"); 
		else {
			for(i=0;i<65536;i++){        
				fscanf(fp,"%d %d\n",&field1[i], &field2[i]);
	  	}
	  	fclose(fp);
		}
	  	
		// your codes begin
		
			dim3 dimGrid(grid_dim,1);
			dim3 dimBlock(block_dim,1);
			
			int threadsize = 65536/(grid_dim * block_dim);
			if (threadsize * grid_dim * block_dim < 65536){
				threadsize++;
			}
			int *gpu_tree1, *gpu_tree2, *gpu_field1, *gpu_field2;
			unsigned long long int *gpu_bv1, *gpu_bv2, *gpu_final_bv;
			float time1, time2;
			cudaEvent_t start2, stop2, start1, stop1;

			cudaMalloc((void**)&gpu_tree1, sizeof(int)*15); // malloc for int*
			cudaMalloc((void**)&gpu_tree2, sizeof(int)*15);
			cudaMalloc((void**)&gpu_bv1, sizeof(unsigned long long int)*15);
			cudaMalloc((void**)&gpu_bv2, sizeof(unsigned long long int)*15);
			cudaMalloc((void**)&gpu_final_bv, sizeof(unsigned long long int)*65536);
			cudaMalloc((void**)&gpu_field1, sizeof(int)*65536);
			cudaMalloc((void**)&gpu_field2, sizeof(int)*65536);
			cudaEventCreate(&start1);
			cudaEventCreate(&stop1);
			cudaEventCreate(&start2);
			cudaEventCreate(&stop2);
			
			cudaEventRecord(start1, 0);
			cudaMemcpy(gpu_tree1, Tree1, sizeof(int)*15, cudaMemcpyHostToDevice); // copy memory for int* 
			cudaMemcpy(gpu_tree2, Tree2, sizeof(int)*15, cudaMemcpyHostToDevice);
			cudaMemcpy(gpu_bv1, BV1, sizeof(unsigned long long int)*15, cudaMemcpyHostToDevice);
			cudaMemcpy(gpu_bv2, BV2, sizeof(unsigned long long int)*15, cudaMemcpyHostToDevice);
			cudaMemcpy(gpu_field1, field1, sizeof(int)*65536, cudaMemcpyHostToDevice);
			cudaMemcpy(gpu_field2, field2, sizeof(int)*65536, cudaMemcpyHostToDevice);

			

			cudaEventRecord(start2,0);
			packet_classify<<<dimGrid, dimBlock>>>(gpu_tree1, gpu_tree2, threadsize, gpu_bv1, gpu_bv2, gpu_final_bv, gpu_field1, gpu_field2);	
			cudaEventRecord(stop2,0);
			cudaEventSynchronize(stop2);
			cudaEventElapsedTime(&time2,start2,stop2);
			cudaEventDestroy(start2);
			cudaEventDestroy(stop2);	
			cout<<"Grid Size: "<<grid_dim<<"*1; Block Size: "<<block_dim<<"*1; Thread Size: "<<threadsize<<"; Time for computation: "<<time2<<"ms; ";		
			//printf("Time Elapsed: %f\n", time);
			
			cudaMemcpy(final_bv, gpu_final_bv, sizeof(unsigned long long int)*65536, cudaMemcpyDeviceToHost);

			cudaEventRecord(stop1,0);
			cudaEventSynchronize(stop1);
			cudaEventElapsedTime(&time1,start1,stop1);
			cudaEventDestroy(start1);
			cudaEventDestroy(stop1);

			cout<<"Total Time: "<<time1<<"ms. "<<"Throughput: "<<65536/time2/1000<<" million packets / second."<<endl;
			//for (i=0; i<4096; i++)
			//{
				
			//	printf("%llu ", final_bv[i]);
				
			//}
			//printf("\n ");
			cudaFree(gpu_tree1);
			cudaFree(gpu_tree2);
			cudaFree(gpu_bv1);
			cudaFree(gpu_bv2);
			cudaFree(gpu_final_bv);
		//}
		
		return 0;

		
		
		
}		