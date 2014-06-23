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
    if(argc!=4){
        cout<<"usage ./openflow  *Packet_num   *Grid_dim   *Block_dim   *Grid_dim_merge   *Block_dim_merge"<<endl; 
        return 0;
    }
    int packet_num = atoi(argv[1]);
    int grid_dim = atoi(argv[2]);
    int block_dim = atoi(argv[3]);
    if (grid_dim*block_dim != packet_num*FIELD){
        cout<<"ERROR: Total number of threads in stage 1 must equal packet_num * FIELD"<<endl;
        return 1;
    }
    cout<<"============================ Experiment Starts ============================"<<endl;
/********************************************************
*   Declare data structs
********************************************************/
    Stage1_Data S1;
    Stage2_Data S2;
    S1.packet_num = packet_num;
    S1.block_dim = block_dim;
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
    long int** bv = new long int*[FIELD*(RULE+1)];
        for(int i = 0; i < FIELD*(RULE+1); i++){
            bv[i] = new long int[int_count];
        }
    S2.bv_final = new long int[packet_num*int_count];
    S2.match_result = new int[packet_num * FIELD];
    S2.merge_result_partial = new long int[int_count*packet_num];

    tree_gen(tree, FIELD, RULE);
    header_gen(header, tree, FIELD, packet_num);
    bv_gen(bv, S2.bv_final, packet_num);

/********************************************************
*   Flatten All the 2D Arrays
********************************************************/
    S1.tree = new int[RULE*FIELD];
    S1.header = new int[packet_num*FIELD];
    S2.bv = new long int[FIELD*(RULE+1) * int_count];

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
    for (int i = 0; i < FIELD*(RULE+1); i++){
        for (int j = 0; j < int_count; j++){
            S2.bv[ i * int_count + j] = bv[i][j];
        }
    }

/********************************************************
*   Setup Timers:
*       1. gpu_tree 
*       2. gpu_header 
*       3. gpu_match_result
********************************************************/
    float time1, time2, time3, time4 = 0;
    cudaEvent_t time_search_memcpyH2D_start, time_search_memcpyH2D_stop, 
                time_search_memcpyD2H_start, time_search_memcpyD2H_stop, 
                time_gpu_start, time_gpu_stop;
    struct timespec cpu_merge_start, cpu_merge_stop;

    cudaEventCreate(&time_search_memcpyH2D_start);
    cudaEventCreate(&time_search_memcpyH2D_stop);
    cudaEventCreate(&time_search_memcpyD2H_start);
    cudaEventCreate(&time_search_memcpyD2H_stop);
    cudaEventCreate(&time_gpu_start);
    cudaEventCreate(&time_gpu_stop);

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
    int* gpu_match_result;

    cudaMalloc((void**)&gpu_tree, sizeof(int)*FIELD*RULE);
        cudaCheckErrors("cudaMalloc gpu_tree");
    cudaMalloc((void**)&gpu_header, sizeof(int)*FIELD*packet_num);
        cudaCheckErrors("cudaMalloc gpu_headers");
    cudaMalloc((void**)&gpu_match_result, sizeof(int)*packet_num*FIELD);
        cudaCheckErrors("cudaMalloc gpu_match_result");
    
    cudaEventRecord(time_search_memcpyH2D_start, 0);

    cudaMemcpy(gpu_tree, S1.tree, sizeof(int)*RULE*FIELD, cudaMemcpyHostToDevice);
        cudaCheckErrors("cudaMemcpy gpu_tree");
    cudaMemcpy(gpu_header, S1.header, sizeof(int)*FIELD*packet_num, cudaMemcpyHostToDevice);
        cudaCheckErrors("cudaMemcpy gpu_headers");

    cudaEventRecord(time_search_memcpyH2D_stop, 0);
    cudaEventSynchronize(time_search_memcpyH2D_stop);
    cudaEventElapsedTime(&time1, time_search_memcpyH2D_start, time_search_memcpyH2D_stop);
    cudaEventDestroy(time_search_memcpyH2D_stop);
    cudaEventDestroy(time_search_memcpyH2D_start);
    cout<<endl<<"Number of Packets: "<<packet_num<<endl;
    cout<<endl<<">>>>>>[Stage 1: Search][GPU] "<<"grid_dim: "<<grid_dim<<", block_dim: "<<block_dim<<endl;
    cout<<endl<<"*  1. Time for memcpy H2D: "<<time1<<"ms, Total bytes copied: "<<endl;
    cout<<"         -> Tree: "<< sizeof(int)*RULE*FIELD<<" Bytes"<<endl;
    cout<<"         -> Headers: "<< sizeof(int)*FIELD*packet_num<<" Bytes"<<endl;
    cout<<"         -> Total Memory Copy: "<< sizeof(int)*RULE*FIELD + sizeof(int)*FIELD*packet_num<<" Bytes"<<endl;

/********************************************************
*   Main Packet Classification Process [Search][GPU]
*       1. Function Call
*       2. Timing
*       3. Memory copy back (gpu_bv_final)
********************************************************/
    cudaEventRecord(time_gpu_start, 0);

    packet_classify<<<dimGrid, dimBlock>>>(gpu_tree, gpu_header, gpu_match_result, S1.packet_num, S1.block_dim);

    cudaCheckErrors("Search fail");
    cudaEventRecord(time_gpu_stop, 0);
    cudaEventSynchronize(time_gpu_stop);
    cudaEventElapsedTime(&time2, time_gpu_start, time_gpu_stop);
    cudaEventDestroy(time_gpu_stop);
    cudaEventDestroy(time_gpu_start);
    cout<<endl<<"*  2. Time for GPU computation: "<<time2<<"ms, GPU throughput: "<<packet_num/time2/1000<<" MPPS"<<endl;


    cudaEventRecord(time_search_memcpyD2H_start, 0);

    cudaMemcpy(S2.match_result, gpu_match_result, sizeof(int)*packet_num*FIELD, cudaMemcpyDeviceToHost);

    cudaEventRecord(time_search_memcpyD2H_stop, 0);
    cudaEventSynchronize(time_search_memcpyD2H_stop);
    cudaEventElapsedTime(&time3, time_search_memcpyD2H_start, time_search_memcpyD2H_stop);
    cudaEventDestroy(time_search_memcpyD2H_stop);
    cudaEventDestroy(time_search_memcpyD2H_start);
    cout<<endl<<"*  3. Time for memcpy D2H: "<<time3<<"ms, Total bytes copied: "<<endl;
    cout<<"         -> Match_result: "<< sizeof(int)*packet_num*FIELD<<" Bytes"<<endl<<endl;

/********************************************************
*   Main Packet Classification Process [Merge][CPU]
********************************************************/

    if (clock_gettime(CLOCK_REALTIME, &cpu_merge_start) == -1){
        perror("clock_gettime: cpu_merge_start");
    } 
    for (int i = 0; i < packet_num; i++){
        for (int j = 0; j < int_count; j++){
            S2.bv_final[i*int_count + j] = bv[S2.match_result[i*FIELD+0]][j] &
                                        bv[S2.match_result[i*FIELD+1]][j] &
                                        bv[S2.match_result[i*FIELD+2]][j] &
                                        bv[S2.match_result[i*FIELD+3]][j] &
                                        bv[S2.match_result[i*FIELD+4]][j] &
                                        bv[S2.match_result[i*FIELD+5]][j] &
                                        bv[S2.match_result[i*FIELD+6]][j] &
                                        bv[S2.match_result[i*FIELD+7]][j] &
                                        bv[S2.match_result[i*FIELD+8]][j] &
                                        bv[S2.match_result[i*FIELD+9]][j] &
                                        bv[S2.match_result[i*FIELD+10]][j] &
                                        bv[S2.match_result[i*FIELD+11]][j] &
                                        bv[S2.match_result[i*FIELD+12]][j] &
                                        bv[S2.match_result[i*FIELD+13]][j] &
                                        bv[S2.match_result[i*FIELD+14]][j]; 
        }
    }
    if (clock_gettime(CLOCK_REALTIME, &cpu_merge_stop) == -1){
        perror("clock_gettime: cpu_merge_stop");
    }
    
    time4 += (cpu_merge_stop.tv_sec - cpu_merge_start.tv_sec) * 1000 + 
             (cpu_merge_stop.tv_nsec - cpu_merge_start.tv_nsec) / 1e6;
    cout<<endl<<">>>>>>[Stage 2: Merge][CPU] "<<endl;
    cout<<endl<<"*  1. Time for Merge: "<<time4<<"ms"<<endl;
    cout<<endl<<"*  2. CPU throughput: "<<packet_num/(time4)/1000<<" MPPS"<<endl;
    cout<<endl<<">>>>>> Total throughput: "<<packet_num/(time1 + time2 + time3 + time4)/1000<<" MPPS"<<endl;
/********************************************************
*   Clear Memory:
*       1. Dynamic allocations on host
*       2. cudaFrees
********************************************************/
    cudaFree(gpu_tree);
    cudaCheckErrors("Free gpu_tree fail");
    cudaFree(gpu_header);
    cudaCheckErrors("Free gpu_headers fail");
    cudaFree(gpu_match_result);
    cudaCheckErrors("Free gpu_match_result fail");

    for (int i = 0; i < FIELD; i++){
        delete tree[i];
    }
    for(int i = 0; i < FIELD; i++){
        delete header[i];
    }
    for(int i = 0; i < FIELD*(RULE+1); i++){
        delete bv[i];
    }
    
    delete tree;
    delete bv;
    delete header;
    
    delete S2.bv_final;
    delete S2.match_result;
    delete S1.tree;
    delete S1.header;
    delete S2.bv;
    delete S2.merge_result_partial;
    //delete partial_merge_threads;
    //delete final_merge_threads;

     cout<<"============================ Experiment Ends ============================"<<endl;
    return 0;
}