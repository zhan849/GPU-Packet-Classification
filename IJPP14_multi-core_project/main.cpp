#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <map>
#include "func.h"
using namespace std;

int main(int argc, char ** argv){
    if(argc!=3)
      { cout<<"usage ./openflow   *Number_of_rules*   * Number_of_concurrent_processing_packets"<<endl; return 0;}
    
    int NUM_RULES = atoi (argv[1]);
    int BATCH = atoi(argv[2]); 
    int RANGE_THREAD = 4*BATCH;
		int EXACT_THREAD = 5*BATCH;
    int NUM_THREAD = 9*BATCH;
    
    int i,j,k,thread_index;
    int rule_size = NUM_RULES;  // the total number of rules      
    int eth_sr_uno = rule_size * 0.25;     
    int eth_dr_uno = rule_size * 0.25;
    int meta_uno 	 = rule_size * 0.1;
		int combine1_uno = rule_size * 0.4;
		int combine2_uno = rule_size * 0.4;
    
    int sip_lb = rule_size * 0.2;
    int sip_ub = rule_size * 0.2;
    int dip_lb = rule_size * 0.2;
    int dip_ub = rule_size * 0.2;
    
    int sp_lb = rule_size * 0.1;
    int sp_ub = rule_size * 0.1;
    int dp_lb = rule_size * 0.1;
    int dp_ub = rule_size * 0.1;
    
    // generate rule seeds
    
    long * eth_sr_seed = generate_seeds (&eth_sr_uno, 48);
    long * eth_dr_seed = generate_seeds (&eth_dr_uno, 48);
    long * meta_seed = generate_seeds (&meta_uno,32);
    long * combine1_seed = generate_seeds (&combine1_uno,32);
    long * combine2_seed = generate_seeds (&combine2_uno,36);

    long * sip_lb_seed = generate_seeds (&sip_lb, 32);
    long * sip_ub_seed = generate_seeds (&sip_ub, 32);
    long * dip_lb_seed = generate_seeds (&dip_lb, 32);
    long * dip_ub_seed = generate_seeds (&dip_ub, 32);
    
    long * sp_lb_seed = generate_seeds (&sp_lb, 16);
    long * sp_ub_seed = generate_seeds (&sp_ub, 16);
    long * dp_lb_seed = generate_seeds (&dp_lb, 16);
    long * dp_ub_seed = generate_seeds (&dp_ub, 16); 
    
    // exact match
    long * eth_sr_rule = generate_rules (eth_sr_seed, eth_sr_uno, rule_size, 48);
    long * eth_dr_rule = generate_rules (eth_dr_seed, eth_dr_uno, rule_size, 48);
    long * meta_rule 	 = generate_rules (meta_seed, meta_uno, rule_size, 32);
		long * combine1_rule = generate_rules (combine1_seed, combine1_uno, rule_size, 32);
		long * combine2_rule = generate_rules (combine1_seed, combine1_uno, rule_size, 36);
    
    // range match
    long * sip_lb_rule = generate_rules (sip_lb_seed, sip_lb, rule_size, 32);
    long * sip_ub_rule = generate_rules (sip_ub_seed, sip_ub, rule_size, 32);
    long * dip_lb_rule = generate_rules (dip_lb_seed, dip_lb, rule_size, 32);
    long * dip_ub_rule = generate_rules (dip_ub_seed, dip_ub, rule_size, 32);
    
    long * sp_lb_rule = generate_rules (sp_lb_seed, sp_lb, rule_size, 16);
    long * sp_ub_rule = generate_rules (sp_ub_seed, sp_ub, rule_size, 16);
    long * dp_lb_rule = generate_rules (dp_lb_seed, dp_lb, rule_size, 16);
    long * dp_ub_rule = generate_rules (dp_ub_seed, dp_ub, rule_size, 16); 
    
    lb_ub (sip_lb_rule, sip_ub_rule, rule_size);
    lb_ub (dip_lb_rule, dip_ub_rule, rule_size);
    lb_ub (sp_lb_rule, sp_ub_rule, rule_size);
    lb_ub (dp_lb_rule, dp_ub_rule, rule_size);
    
    int sip_pool_size = sip_lb+sip_ub;
    int dip_pool_size = dip_lb+dip_ub;
    int sp_pool_size = sp_lb+sp_ub;
    int dp_pool_size = dp_lb+dp_ub;
    
    // merge the ub and lb
    long * sip_pool = (long*) malloc( sizeof(long)* sip_pool_size); 
    long * dip_pool = (long*) malloc( sizeof(long)* dip_pool_size);
    long * sp_pool = (long*) malloc( sizeof(long)* sp_pool_size);
    long * dp_pool = (long*) malloc( sizeof(long)* dp_pool_size);
    
    for (i=0; i< sip_lb; i++){	
        sip_pool[i] = sip_lb_seed[i];
        dip_pool[i] = dip_lb_seed[i];
    }
    
    for (i=0; i< sip_ub; i++){	
        sip_pool[i+sip_lb] = sip_ub_seed[i];
        dip_pool[i+dip_lb] = dip_ub_seed[i];
    }
    
    for (i=0; i< sp_lb; i++){	
        sp_pool[i] = sp_lb_seed[i];
        dp_pool[i] = dp_lb_seed[i];
    }
    
    for (i=0; i< sp_ub; i++){	
        sp_pool[i+sp_lb] = sp_ub_seed[i];
        dp_pool[i+dp_lb] = dp_ub_seed[i];
    }
    
    // Generate the trace 
    long * EthSr = (long *) malloc (sizeof(long) * NUM_PACKET);
    for(i = 0; i< NUM_PACKET; i++)	
    	EthSr[i] = eth_sr_seed[rand() % eth_sr_uno];
    
    long * EthDr = (long *) malloc (sizeof(long) * NUM_PACKET);
    for(i = 0; i< NUM_PACKET; i++)	
    	EthDr[i] = eth_dr_seed[rand() % eth_dr_uno];
    
    long * Meta = (long *) malloc (sizeof(long) * NUM_PACKET);
    for(i = 0; i< NUM_PACKET; i++)	
    	Meta[i] = meta_seed[rand() % meta_uno];
    
    long * Combine1 = (long *) malloc (sizeof(long) * NUM_PACKET);
    for(i = 0; i< NUM_PACKET; i++)	
    	Combine1[i] = combine1_seed[rand() % combine1_uno];
    
    long * Combine2 = (long *) malloc (sizeof(long) * NUM_PACKET);
    for(i = 0; i< NUM_PACKET; i++)	
    	Combine2[i] = combine2_seed[rand() % combine2_uno];
    
    long * Sip = (long *) malloc (sizeof(long) * NUM_PACKET);
    for(i = 0; i< NUM_PACKET; i++)	
    	Sip[i] = sip_pool[rand() % sip_pool_size];
    
    long * Dip = (long *) malloc (sizeof(long) * NUM_PACKET);
    for(i = 0; i< NUM_PACKET; i++)	
    	Dip[i] = dip_pool[rand() % dip_pool_size];
    
    long * Sp = (long *) malloc (sizeof(long) * NUM_PACKET);
    for(i = 0; i< NUM_PACKET; i++)	
    	Sp[i] = sp_pool[rand() % sp_pool_size];
    
    long * Dp = (long *) malloc (sizeof(long) * NUM_PACKET);
    for(i = 0; i< NUM_PACKET; i++)	
    	Dp[i] = dp_pool[rand() % dp_pool_size];
    	 
    Point * EthSr_Points = ConstructTree_points(eth_sr_seed, eth_sr_uno, eth_sr_rule, rule_size); 
    Point * EthDr_Points = ConstructTree_points(eth_dr_seed, eth_dr_uno, eth_dr_rule, rule_size); 
    Point * Meta_Points = ConstructTree_points(meta_seed, meta_uno, meta_rule, rule_size); 
    Point * Combine1_Points = ConstructTree_points(combine1_seed, combine1_uno, combine1_rule, rule_size); 
    Point * Combine2_Points = ConstructTree_points(combine2_seed, combine2_uno, combine2_rule, rule_size); 
    
    sip_pool_size = trim_bound(sip_pool, 0 , sip_pool_size-1);
    Set * sipSet = new Set[sip_pool_size+1];
		for(i=1; i<=sip_pool_size;i++){
			int no = 0;
			for(j=0; j< NUM_RULES-1; j++){
				  if (sip_pool[i-1] >= sip_lb_rule[j] && sip_pool[i]<= sip_ub_rule[j]){
							sipSet[i].size ++;
				  }	
			}
			int * set = new int[sipSet[i].size];
			sipSet[i].size = no;
			for(j=0; j< NUM_RULES-1; j++){
				  if (sip_pool[i-1] >= sip_lb_rule[j] && sip_pool[i]<= sip_ub_rule[j]){
							set[no] = j;
							no++;
				  }	
			}
			sipSet[i].element = set;
			free(set);
		}
    long * sipArray = new long[sip_pool_size];
		Gen_tree_2(sipArray, 0, sip_pool_size-1, sip_pool, 0, sip_pool_size-1);
		free(sip_pool);
    
    dip_pool_size = trim_bound(dip_pool, 0 , dip_pool_size-1);
    Set * dipSet = new Set[dip_pool_size+1];
		for(i=1; i<=dip_pool_size;i++){
			int no = 0;
			for(j=0; j< NUM_RULES-1; j++){
				  if (dip_pool[i-1] >= dip_lb_rule[j] && dip_pool[i]<= dip_ub_rule[j]){
							dipSet[i].size ++;
				  }	
			}
			int * set = new int[dipSet[i].size];
			dipSet[i].size = no;
			for(j=0; j< NUM_RULES-1; j++){
				  if (dip_pool[i-1] >= dip_lb_rule[j] && dip_pool[i]<= dip_ub_rule[j]){
							set[no] = j;
							no++;
				  }	
			}
			dipSet[i].element = set;
			free(set);
		}
    long * dipArray = new long[dip_pool_size];
		Gen_tree_2(dipArray, 0, dip_pool_size-1, dip_pool, 0, dip_pool_size-1);
		free(dip_pool);
    	
    sp_pool_size = trim_bound(sp_pool, 0 , sp_pool_size-1);
    Set * spSet = new Set[sp_pool_size+1];
		for(i=1; i<=sp_pool_size;i++){
			int no = 0;
			for(j=0; j< NUM_RULES-1; j++){
				  if (sp_pool[i-1] >= sp_lb_rule[j] && sp_pool[i]<= sp_ub_rule[j]){
							spSet[i].size ++;
				  }	
			}
			int * set = new int[spSet[i].size];
			spSet[i].size = no;
			for(j=0; j< NUM_RULES-1; j++){
				  if (sp_pool[i-1] >= sp_lb_rule[j] && sp_pool[i]<= sp_ub_rule[j]){
							set[no] = j;
							no++;
				  }	
			}
			spSet[i].element = set;
			free(set);
		}
    long * spArray = new long[sp_pool_size];
		Gen_tree_2(spArray, 0, sp_pool_size-1, sp_pool, 0, sp_pool_size-1);
		free(sp_pool);
		
		dp_pool_size = trim_bound(dp_pool, 0 , dp_pool_size-1);
    Set * dpSet = new Set[dp_pool_size+1];
		for(i=1; i<=dp_pool_size;i++){
			int no = 0;
			for(j=0; j< NUM_RULES-1; j++){
				  if (dp_pool[i-1] >= dp_lb_rule[j] && dp_pool[i]<= dp_ub_rule[j]){
							dpSet[i].size ++;
				  }	
			}
			int * set = new int[dpSet[i].size];
			dpSet[i].size = no;
			for(j=0; j< NUM_RULES-1; j++){
				  if (dp_pool[i-1] >= dp_lb_rule[j] && dp_pool[i]<= dp_ub_rule[j]){
							set[no] = j;
							no++;
				  }	
			}
			dpSet[i].element = set;
			free(set);
		}
    long * dpArray = new long[dp_pool_size];
		Gen_tree_2(dpArray, 0, dp_pool_size-1, dp_pool, 0, dp_pool_size-1);
		free(dp_pool);	
    	  
    
    free(eth_sr_seed);
    free(eth_dr_seed);
    free(meta_seed);
    free(combine1_seed);
    free(combine2_seed);
    free(sip_lb_seed);
    free(sip_ub_seed);
    free(dip_lb_seed);
    free(dip_ub_seed);
    free(sp_lb_seed);
    free(sp_ub_seed);
    free(dp_lb_seed);
    free(dp_ub_seed);
    
    ////built up hash table
    cout<<"build hash table"<<endl;
    int hash_table1_size =  combine1_uno * 2;
    int hash_table2_size =  combine2_uno * 2;
    int hash_table3_size =  meta_uno * 2;
    int hash_table4_size =  eth_sr_uno * 2;
    int hash_table5_size =  eth_dr_uno * 2;
    
    int hash_fun1_key = combine1_uno * 2;
    int hash_fun2_key = combine2_uno * 2;
    int hash_fun3_key = meta_uno * 2;
    int hash_fun4_key = eth_sr_uno * 2;
    int hash_fun5_key = eth_dr_uno * 2;
    
    hash_slot * hash_table1 = new hash_slot[hash_table1_size];
    hash_slot * hash_table2 = new hash_slot[hash_table2_size];
    hash_slot * hash_table3 = new hash_slot[hash_table3_size];
    hash_slot * hash_table4 = new hash_slot[hash_table4_size];
    hash_slot * hash_table5 = new hash_slot[hash_table5_size];
    
    int success;

		success = generate_hash_table(hash_table3, hash_table3_size, Meta_Points, meta_uno, hash_fun3_key);
    while (!success) {
        for (i=0; i<hash_fun3_key; i++) { // clear hash table
            hash_table3[i].key = -1;
            hash_table3[i].set = NULL;
        }
        hash_fun3_key--; // reconfig hash function
        success = generate_hash_table(hash_table3, hash_table3_size, Meta_Points, meta_uno, hash_fun3_key);
    }
    
    success = generate_hash_table(hash_table4, hash_table4_size, EthSr_Points, eth_sr_uno, hash_fun4_key);
    while (!success) {
        for (i=0; i<hash_fun4_key; i++) { // clear hash table
            hash_table4[i].key = -1;
            hash_table4[i].set = NULL;
        }
        hash_fun4_key--; // reconfig hash function
        success = generate_hash_table(hash_table4, hash_table4_size, EthSr_Points, eth_sr_uno, hash_fun4_key);
    }
		BATCH*=1.5;
		success = generate_hash_table(hash_table5, hash_table5_size, EthDr_Points, eth_dr_uno, hash_fun5_key);
    while (!success) {
        for (i=0; i<hash_fun5_key; i++) { // clear hash table
            hash_table5[i].key = -1;
            hash_table5[i].set = NULL;
        }
        hash_fun5_key--; // reconfig hash function
        success = generate_hash_table(hash_table5, hash_table5_size, EthDr_Points, eth_dr_uno, hash_fun5_key);
    }
    
    success = generate_hash_table(hash_table2, hash_table2_size, Combine2_Points, combine2_uno, hash_fun2_key);   
    while (!success) {
        for (i=0; i<hash_fun2_key; i++) { // clear hash table
            hash_table2[i].key = -1;
            hash_table2[i].set = NULL;
        }
        hash_fun2_key--; // reconfig hash function
        success = generate_hash_table(hash_table2, hash_table2_size, Combine2_Points, combine2_uno, hash_fun2_key);
    }
    
    success = generate_hash_table(hash_table1, hash_table1_size, Combine1_Points, combine1_uno, hash_fun1_key);   
    while (!success) {
        for (i=0; i<hash_fun1_key; i++) { // clear hash table
            hash_table1[i].key = -1;
            hash_table1[i].set = NULL;
        }
        hash_fun1_key--; // reconfig hash function
        success = generate_hash_table(hash_table1, hash_table1_size, Combine1_Points, combine1_uno, hash_fun1_key);
    }
    
    cout<< "finish setting up hash tables"<<endl;
           
    free(eth_sr_rule);
    free(eth_dr_rule);
    free(meta_rule);
    free(combine1_rule);
    free(combine2_rule);
    free(sip_lb_rule);
    free(sip_ub_rule);
    free(dip_lb_rule);
    free(dip_ub_rule);
    free(sp_lb_rule);
    free(sp_ub_rule);
    free(dp_lb_rule);
    free(dp_ub_rule);
    
    Set * sipSets[BATCH];
		Set * dipSets[BATCH];
		Set * spSets [BATCH];
		Set * dpSets [BATCH];
		Set * combine1[BATCH];
		Set * combine2[BATCH];
		Set * metaSets[BATCH];
		Set * esrcSets[BATCH];
		Set * edesSets[BATCH];
		
		for (i=0; i< BATCH; i++){
			sipSets[i] = new Set[TRACE];
			dipSets[i] = new Set[TRACE];
			spSets[i]  = new Set[TRACE];
			dpSets[i]  = new Set[TRACE];
			combine1[i]= new Set[TRACE];
			combine2[i]= new Set[TRACE];
			metaSets[i]= new Set[TRACE];
			esrcSets[i]= new Set[TRACE];
			edesSets[i]= new Set[TRACE];
		}
		
		pthread_attr_t attr;
		pthread_t* threads = (pthread_t*) malloc( sizeof(pthread_t) * RANGE_THREAD);
		pthread_t* threads_H = (pthread_t*) malloc( sizeof(pthread_t) * EXACT_THREAD);
		pthread_t* Mthreads = (pthread_t*) malloc( sizeof(pthread_t) * BATCH);
		pthread_attr_init(&attr);
		pthread_attr_setscope(&attr,PTHREAD_SCOPE_SYSTEM);
		thread_param_t * thread_params = (thread_param_t*) malloc( sizeof(thread_param_t) * RANGE_THREAD );	
		thread_param_t_H * thread_params_H = (thread_param_t_H*) malloc( sizeof(thread_param_t_H) * EXACT_THREAD);
		merge_thread_param_t * merge_thread_params = (merge_thread_param_t*) malloc(sizeof(merge_thread_param_t)*BATCH);
		
		for(thread_index=0; thread_index<RANGE_THREAD; thread_index++) {
			thread_params[thread_index].thread_id = thread_index;
			switch(thread_index%4){
				case 0:
					thread_params[thread_index].output = sipSets[thread_index/4];
					thread_params[thread_index].TreeArray = sipArray;
					thread_params[thread_index].TreeSize = sip_pool_size;
					thread_params[thread_index].input = Sip;
					thread_params[thread_index].Lists = sipSet;
					break;
				case 1:
					thread_params[thread_index].output = dipSets[thread_index/4];
					thread_params[thread_index].TreeArray = dipArray;
					thread_params[thread_index].TreeSize = dip_pool_size;
					thread_params[thread_index].input = Dip;
					thread_params[thread_index].Lists = dipSet;
					break;
				case 2:
					thread_params[thread_index].output = spSets[thread_index/4];
					thread_params[thread_index].TreeArray = spArray;
					thread_params[thread_index].TreeSize = sp_pool_size;
					thread_params[thread_index].input = Sp;
					thread_params[thread_index].Lists = spSet;
					break;
				case 3:
					thread_params[thread_index].output = dpSets[thread_index/4];
					thread_params[thread_index].TreeArray = dpArray;
					thread_params[thread_index].TreeSize = dp_pool_size;
					thread_params[thread_index].input = Dp;
					thread_params[thread_index].Lists = dpSet;
					break;
				default:printf("invalid thread_index");	
			}
  	}
    for(thread_index=0; thread_index<EXACT_THREAD; thread_index++) {
        switch(thread_index%5){
            case 0:
                thread_params_H[thread_index].Table = hash_table1;
                thread_params_H[thread_index].trace = Combine1;
                thread_params_H[thread_index].hash_function_key = hash_fun1_key;
                thread_params_H[thread_index].output = combine1[thread_index/5];
                break;
            case 1:
                thread_params_H[thread_index].Table = hash_table2;
                thread_params_H[thread_index].trace = Combine2;
                thread_params_H[thread_index].hash_function_key = hash_fun2_key;
								thread_params_H[thread_index].output = combine2[thread_index/5];
                break;
            case 2:
                thread_params_H[thread_index].Table = hash_table3;
                thread_params_H[thread_index].trace = Meta;
                thread_params_H[thread_index].hash_function_key = hash_fun3_key;
								thread_params_H[thread_index].output = metaSets[thread_index/5];
                break;
            case 3:
                thread_params_H[thread_index].Table = hash_table4;
                thread_params_H[thread_index].trace = EthSr;
                thread_params_H[thread_index].hash_function_key = hash_fun4_key;
								thread_params_H[thread_index].output = esrcSets[thread_index/5];
                break;
            case 4:
                thread_params_H[thread_index].Table = hash_table2;
                thread_params_H[thread_index].trace = EthDr;
                thread_params_H[thread_index].hash_function_key = hash_fun2_key;
								thread_params_H[thread_index].output = edesSets[thread_index/5];
                break;    
            default :printf("invalid thread_index");				
        }
  	}
  	
  	for(thread_index=0; thread_index<BATCH ; thread_index++) {
        merge_thread_params[thread_index].Set1 =sipSets[thread_index];
        merge_thread_params[thread_index].Set2 =dipSets[thread_index];
        merge_thread_params[thread_index].Set3 =spSets[thread_index];
        merge_thread_params[thread_index].Set4 =dpSets[thread_index];
        merge_thread_params[thread_index].Set5 =combine1[thread_index];
        merge_thread_params[thread_index].Set6 =combine2[thread_index];
        merge_thread_params[thread_index].Set7 =metaSets[thread_index];
        merge_thread_params[thread_index].Set8 =esrcSets[thread_index];
        merge_thread_params[thread_index].Set9 =edesSets[thread_index];
    }
 		struct timespec start, stop, stop1;
		double t = 0;
		
		if( clock_gettime( CLOCK_REALTIME, &start) == -1 )  perror( "clock gettime" );	
        
		for(thread_index=0; thread_index<RANGE_THREAD; thread_index++) {
			if(pthread_create(&threads[thread_index], &attr, (void*(*)(void *))Generate_BV, (void*) &thread_params[thread_index])!=0){
				printf("Creating Thread failed!\n");}
		}			
		
		for(thread_index=0; thread_index<EXACT_THREAD; thread_index++) {
    	if(pthread_create(&threads_H[thread_index], &attr, (void*(*)(void *))Lookup_Hash, (void*) &thread_params_H[thread_index])!=0){
            printf("Creating Thread failed!\n");
    	}
    }
    

    ///// join all the threads
		for(thread_index=0; thread_index<RANGE_THREAD; thread_index++) {
			if (pthread_join(threads[thread_index], NULL)!=0){
				printf("Joining thread failed!\n");}  
		}
		for(thread_index=0; thread_index<EXACT_THREAD; thread_index++) {
	    if (pthread_join(threads_H[thread_index], NULL)!=0){
        printf("Joining thread failed!\n");
	  	}  
    }  

		if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}
		t =t+( stop.tv_sec - start.tv_sec ) + (double)( stop.tv_nsec - start.tv_nsec )/1e9;
		
		printf("*** throughput is %f MLPS ***\n", NUM_PACKET*BATCH/(t*1e6-BATCH*1e2));
		printf("*** latency is %f nsec ***\n", t);
		
    //free trace
    free(threads);
    free(threads_H);
		free(thread_params);
		free(thread_params_H);
		free(Mthreads);
		free(merge_thread_params);
    
    free(EthSr);
    free(EthDr);
    free(Meta);
    free(Sip);
    free(Dip);
    free(Sp);
    free(Dp);
    free(Combine1);
    free(Combine2);
    
    free(EthSr_Points);
    free(EthDr_Points);
    free(Combine1_Points);
    free(Combine2_Points);
    free(Meta_Points);
 		
 		free(sipArray);
 		free(dipArray);
 		free(spArray);
 		free(dpArray);
    
   	for (i=0; i<BATCH; i++){
			free(sipSets[i]);
			free(dipSets[i]);
			free(spSets[i]);
			free(dpSets[i]);
			free(combine1[i]);
			free(combine2[i]);
			free(metaSets[i]);
			free(esrcSets[i]);
			free(edesSets[i]);
   	}
        
    return 0;
}
