#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cublas.h>
using namespace std;

int main(){
	// Get Device Property
 int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    cout<<"GPU Device Number: "<<i<<endl;
    cout<<"Device Name: "<<prop.name<<endl;
    cout<<"Number of Multi-Processor: "<<prop.multiProcessorCount<<endl;
    cout<<"Clock Rate(kHZ): "<<prop.clockRate<<endl;
    cout<<endl;

    cout<<"Registers Per Block: "<<prop.regsPerBlock<<endl;
    cout<<"Shared Memory Per Block(Bytes): "<<prop.sharedMemPerBlock<<endl;
    cout<<"L2 Cache Size(Bytes): "<<prop.l2CacheSize<<endl;
    cout<<endl;

    cout<<"Memory Clock Rate(KHz): "<<prop.memoryClockRate <<endl;
    cout<<"Total Global Memory(Bytes): "<<prop.totalGlobalMem<<endl;
    cout<<"Total Constant Memory(Bytes): "<<prop.totalConstMem<<endl;
    cout<<"Memory Bus Width(Bits): "<<prop.memoryBusWidth<<endl;
    cout<<"Maximum pitch allowed by memory copies(Bytes): "<<prop.memPitch<<endl;
    cout<<endl;
    
    cout<<"Max Thread Per Block: "<<prop.maxThreadsPerBlock<<endl;
    cout<<"Max Dimension of a Block: "<<prop.maxThreadsDim[0]<<" * "<<prop.maxThreadsDim[1]<<" * "<<prop.maxThreadsDim[2]<<endl;
    cout<<"Max Dimension of a Grid: "<<prop.maxGridSize[0]<<" * "<<prop.maxGridSize[1]<<" * "<<prop.maxGridSize[2]<<endl;
    cout<<"Max Threads per Multi-Processor: "<<prop.maxThreadsPerMultiProcessor<<endl;
    cout<<endl;

    cout<<"Max Texture 1D: "<<prop.maxTexture1D<<endl;
    cout<<"Max Texture 1D bounded to linear memory: "<<prop.maxTexture1DLinear<<endl;
    cout<<"Max Texture 2D: "<<prop.maxTexture2D[0]<<" * "<<prop.maxTexture2D[1]<<endl;
    cout<<"Max Texture 2D bounded to pitched memory(width * height * pitch): "<<prop.maxTexture2DLinear[0]<<" * "<<prop.maxTexture2DLinear[1]<<" * "<<prop.maxTexture2DLinear[2]<<endl;
    cout<<endl;
    cout<<"========================================================"<<endl;
  }

  return 0;
}
/*
 int maxTexture1D;
        int maxTexture1DLinear;
        int maxTexture2D[2];
        int maxTexture2DLinear[3];



*/