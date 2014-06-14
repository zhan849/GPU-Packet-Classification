#include <stdlib.h>
#include <stdio.h>
#include <time.h>
//execution time: 16.612064 ms
#define n 1024
#define block_size 32
__global__ void mul_matrix(int *a, int *b, int *c){
	int my_x, my_y, i,j;
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	int row = threadIdx.y;
	int col = threadIdx.x;

	
	my_y = blockIdx.y*blockDim.y + threadIdx.y;
	my_x = blockIdx.x*blockDim.x + threadIdx.x;	
	
	int local_c;
	__shared__ int A_s[32][32];
	__shared__ int B_s[32][32];
	
	for (i=0;i<n/block_size;i++)
	{
		A_s[row][col] = a[my_x*n + (i*blockDim.x + col)];
		B_s[row][col] = b[(i*blockDim.y + row)*n + my_y];
	
	
		__syncthreads();
		
		for(j=0;j<block_size;j++)
			local_c += A_s[row][j]*B_s[j][col];
		__syncthreads();
	}
	c[my_x*n+my_y] = local_c;	 
}

int main(){		
    int i;
	float time;
	cudaEvent_t start, stop;
    // row major order
    // a(i,j) = a[i*1024+j];
    int *a = (int*)malloc(sizeof(int)*n*n);           
		int *b = (int*)malloc(sizeof(int)*n*n); 
		int *c = (int*)malloc(sizeof(int)*n*n);           
    
   	dim3 dimGrid(32,32);
		dim3 dimBlock(32,32);
		
		for (i=0; i<n*n; i++){
			a[i]=1;
			b[i]=2;
			c[i]=0;
		}	 
		int *gpu_a, *gpu_b, *gpu_c;
		cudaMalloc((void**)&gpu_a, sizeof(int)*n*n); 
		cudaMalloc((void**)&gpu_b, sizeof(int)*n*n);
		cudaMalloc((void**)&gpu_c, sizeof(int)*n*n);
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		
		cudaMemcpy(gpu_a, a, sizeof(int)*n*n, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_b, b, sizeof(int)*n*n, cudaMemcpyHostToDevice);
		
		cudaEventRecord(start,0);		
		mul_matrix<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c);				
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		
		cudaMemcpy(c, gpu_c, sizeof(int)*n*n, cudaMemcpyDeviceToHost);
  	
  	//for (i=0; i<n*n; i++)
	//printf("%d ", c[i]);
  		printf("%f ", time);
  		
		free(a);
		free(b);
		free(c);
  	cudaFree(gpu_a);  
		cudaFree(gpu_b);  
		cudaFree(gpu_c);  
		return 0;
}	
