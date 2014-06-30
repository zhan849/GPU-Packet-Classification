#include <iostream>
#include <pthread.h>
#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cublas.h>
#include "gpu_pc_v2_func.h"

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


int main(int argc, char** argv){
    if(argc!=3){
        cout<<"usage ./openflow  *Packet_num   *Grid_dim   *Block_dim   *BATCH"<<endl; 
        return 0;
    }
    int packet_num = atoi(argv[1]);
    int grid_dim = atoi(argv[2]);
    //int block_dim = atoi(argv[3]);
    //int BATCH = atoi(argv[4]);
    if (grid_dim*block_dim != packet_num){
        cout<<"ERROR: Total number of threads in stage 1 must equal packet_num"<<endl;
        return 1;
    }
    cout<<"============================ Experiment Starts ============================"<<endl;
/********************************************************
*   Declare data structs
********************************************************/
    Stage1_Data S1;
    Stage2_Data S2;
    S1.packet_num = packet_num;
/********************************************************
*   Preparing Data:
*       1. Generate random header
*       2. Generate BVs 
*       3. Generate random packets
*       4. Deliberately make some rule-matching packets
********************************************************/
    srand(time(NULL));
    int** tree = new int*[FIELD];
        for(int i = 0; i < FIELD; i++){
            tree[i] = new int[RULE];
        }
    int** header = new int*[FIELD];
        for (int i = 0; i < FIELD; i++){
            header[i] = new int[packet_num];
        }
    /*long int** bv = new long int*[FIELD*(RULE+1)];
        for(int i = 0; i < FIELD*(RULE+1); i++){
            bv[i] = new long int[int_count];
        }*/
    int* bv = new int[FIELD * (RULE + 1) * int_count];
    int* bv_final = new int[packet_num * int_count];
    //S2.merge_result_partial = new long int[int_count*packet_num];

    tree_gen(tree, FIELD, RULE);
    header_gen(header, tree, FIELD, packet_num);
    bv_gen_short(bv, bv_final, packet_num);

/********************************************************
*   Flatten All the 2D Arrays
********************************************************/
    S1.tree = new int[RULE*FIELD];
    S1.header = new int[packet_num*FIELD];
    //S2.bv = new long int[FIELD*(RULE+1) * int_count];

    for (int i = 0; i < FIELD; i++){
        for (int j = 0; j < RULE; j++){
            S1.tree[i*RULE+j] = tree[i][j];
        }
    }
    for (int i = 0; i < FIELD; i++){
        for (int j = 0; j < packet_num; j++){
            S1.header[i*packet_num + j] = header[i][j];
        }
    }
    //for (int i = 0; i < FIELD*(RULE+1); i++){
    //    for (int j = 0; j < int_count; j++){
    //        S2.bv[ i * int_count + j] = bv[i][j];
    //    }
   // }

/********************************************************
*   Setup Timers:
*       1. GPU Timer
*       2. CPU Timer
********************************************************/
    float time1 = 0, time2 = 0, time3 = 0;
    cudaEvent_t time_memcpyH2D_start, time_memcpyH2D_stop, 
                    time_pc_start, time_pc_stop, 
                    time_memcpyD2H_start, time_memcpyD2H_stop;
    
    cudaEventCreate(&time_memcpyH2D_start);
    cudaEventCreate(&time_memcpyH2D_stop);
    cudaEventCreate(&time_pc_start);
    cudaEventCreate(&time_pc_stop);
    cudaEventCreate(&time_memcpyD2H_start);
    cudaEventCreate(&time_memcpyD2H_stop);

/********************************************************
*   Allocate Space in Device:
*       1. gpu_tree 
*       2. gpu_header 
*       3. gpu_match_result
********************************************************/
    dim3 dimGrid(grid_dim,1);
    dim3 dimBlock(block_dim,1);

    int* gpu_tree;
    int* gpu_header;
    int* gpu_bv;
    int* gpu_bv_final;

    cudaMalloc((void**)&gpu_tree, sizeof(int)*FIELD*RULE);
        cudaCheckErrors("cudaMalloc gpu_tree");
    cudaMalloc((void**)&gpu_header, sizeof(int)*FIELD*packet_num);
        cudaCheckErrors("cudaMalloc gpu_headers");
    cudaMalloc((void**)&gpu_bv, sizeof(int) * FIELD * (RULE + 1) * int_count);
        cudaCheckErrors("cudaMalloc gpu_bv");
    cudaMalloc((void**)&gpu_bv_final, sizeof(int) * packet_num * int_count);
        cudaCheckErrors("cudaMalloc gpu_bv_final");
    
    cudaEventRecord(time_memcpyH2D_start, 0);

    cudaMemcpy(gpu_tree, S1.tree, sizeof(int)*RULE*FIELD, cudaMemcpyHostToDevice);
        cudaCheckErrors("cudaMemcpy gpu_tree");
    cudaMemcpy(gpu_header, S1.header, sizeof(int)*FIELD*packet_num, cudaMemcpyHostToDevice);
        cudaCheckErrors("cudaMemcpy gpu_headers");
    cudaMemcpy(gpu_bv, bv, sizeof(int) * FIELD * (RULE + 1) * int_count, cudaMemcpyHostToDevice);
        cudaCheckErrors("cudaMemcpy gpu_bv");
//    cudaMemcpy(gpu_bv_final, bv_final, sizeof(int) * packet_num * int_count, cudaMemcpyHostToDevice);
//        cudaCheckErrors("cudaMemcpy gpu_bv");

    cudaEventRecord(time_memcpyH2D_stop, 0);
    cudaEventSynchronize(time_memcpyH2D_stop);
    cudaEventElapsedTime(&time1, time_memcpyH2D_start, time_memcpyH2D_stop);
    cudaEventDestroy(time_memcpyH2D_stop);
    cudaEventDestroy(time_memcpyH2D_start);
    cout<<endl<<"Number of Packets: "<<packet_num<<", Int count: "<<int_count<<endl;
    cout<<endl<<">>>>>>[Stage 1: Search][GPU] "<<"grid_dim: "<<grid_dim<<", block_dim: "<<block_dim<<endl;
    cout<<endl<<"*  1. Time for memcpy H2D: "<<time1<<"ms, Total bytes copied: "<<endl;
    cout<<"         -> Tree: "<< sizeof(int)*RULE*FIELD<<" Bytes"<<endl;
    cout<<"         -> Headers: "<< sizeof(int)*FIELD*packet_num<<" Bytes"<<endl;
    cout<<"         -> BV: "<< sizeof(int) * FIELD * (RULE + 1) * int_count <<" Bytes"<<endl;   
    //cout<<"         -> BV_final: "<< sizeof(int) * packet_num * int_count <<" Bytes"<<endl;   
    cout<<"         -> Total Memory Copy: "<< sizeof(int)*RULE*FIELD + sizeof(int)*FIELD*packet_num + sizeof(int) * FIELD * (RULE + 1) * int_count <<" Bytes"<<endl;

/********************************************************
*   Main Packet Classification Process [Search][GPU]
*       1. Function Call
*       2. Timing
*       3. Memory copy back (gpu_bv_final)
********************************************************/
    cudaEventRecord(time_pc_start, 0);

    pc_short<<<dimGrid, dimBlock>>>(gpu_tree, gpu_header, gpu_bv, gpu_bv_final, packet_num);
    //cudaCheckErrors("packet_classify");
    cudaThreadSynchronize();

    cudaCheckErrors("Search fail");
    cudaEventRecord(time_pc_stop, 0);
    cudaEventSynchronize(time_pc_stop);
    cudaEventElapsedTime(&time2, time_pc_start, time_pc_stop);
    cudaEventDestroy(time_pc_stop);
    cudaEventDestroy(time_pc_start);
    cout<<endl<<"*  2. Time for GPU computation: "<<time2<<"ms"<<endl;
    cout<<"         -> Shared Memory Usage: "<< sizeof(int)*RULE*FIELD + sizeof(int)*FIELD*(RULE+1)*int_count + sizeof(int) * block_dim<<" Bytes / 49152 Bytes"<<endl;

    cudaEventRecord(time_memcpyD2H_start, 0);

    cudaMemcpy(bv_final, gpu_bv_final, sizeof(int)*packet_num*int_count, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy bv_final D2H");
    cudaEventRecord(time_memcpyD2H_stop, 0);
    cudaEventSynchronize(time_memcpyD2H_stop);
    cudaEventElapsedTime(&time3, time_memcpyD2H_start, time_memcpyD2H_stop);
    cudaEventDestroy(time_memcpyD2H_stop);
    cudaEventDestroy(time_memcpyD2H_start);
    cout<<endl<<"*  3. Time for memcpy D2H: "<<time3<<"ms, Total bytes copied: "<<endl;
    cout<<"         -> bv_final: "<< sizeof(int)*packet_num*int_count<<" Bytes"<<endl<<endl;



    cout<<endl<<">>>>>> Total time for GPU: "<< time1 + time2 + time3<<"ms"<<endl;
    cout<<">>>>>> GPU throughput (Compute Only): "<<packet_num/time2/1000<<" MPPS"<<endl;
    cout<<">>>>>> Total throughput (Including memory copy: "<<packet_num/(time1 + time2 + time3)/1000<<" MPPS"<<endl<<endl;
/********************************************************
*   Clear Memory:
*       1. Dynamic allocations on host
*       2. cudaFrees
********************************************************/
    cudaFree(gpu_tree);
    cudaCheckErrors("Free gpu_tree fail");
    cudaFree(gpu_header);
    cudaCheckErrors("Free gpu_headers fail");
    cudaFree(gpu_bv);
    cudaCheckErrors("Free gpu_bv_final fail");
    cudaFree(gpu_bv_final);
    cudaCheckErrors("Free gpu_bv_final fail");

    for (int i = 0; i < FIELD; i++){
        delete tree[i];
    }
    for(int i = 0; i < FIELD; i++){
        delete header[i];
    }

   /* for (int i = 0; i < packet_num / BATCH; i++){
        delete data_C[i].partial_merge_source;
    }*/
    delete tree;
    delete bv;
    delete header;
    
    //delete S2.bv_final;
    //delete S2.match_result;
    delete S1.tree;
    delete S1.header;
    delete bv_final;
    //delete S2.bv;
    //delete S2.merge_result_partial;
    //delete partial_merge_threads;
    //delete final_merge_threads;

     cout<<"============================ Experiment Ends ============================"<<endl;
    return 0;
}