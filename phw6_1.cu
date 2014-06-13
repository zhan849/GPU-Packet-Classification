#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define n 1024
//execution time 251.549927 ms
__global__ void mul_matrix(int *a, int *b, int *c){
	int my_x, my_y, i;
	my_x = blockIdx.x*blockDim.x + threadIdx.x;	
	my_y = blockIdx.y*blockDim.y + threadIdx.y;
	
	int local_c;
	for (i=0;i<n;i++)
		local_c += a[my_x*n+i]*b[i*n+my_y];
	
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
	//printf("/n ");
  		printf("%f ", time);
  		
		free(a);
		free(b);
		free(c);
  	cudaFree(gpu_a);  
		cudaFree(gpu_b);  
		cudaFree(gpu_c);  
		return 0;
}	
