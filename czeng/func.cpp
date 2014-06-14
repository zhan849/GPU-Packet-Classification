#include "func.h"

long * generate_seeds (int * Seed_size, int bit_length){
    long Rule_Max = (long)pow(2,bit_length);
    int seed_size = *Seed_size;
    seed_size = (seed_size < Rule_Max) ? seed_size : Rule_Max;
    *Seed_size = seed_size;
    long * seed = (long *) malloc (sizeof(long) * seed_size); 
    srand(time(NULL));
    seed[0] = rand() % Rule_Max;
    
    long temp; 
    int j,current_rule_size = 1;
    
    while (current_rule_size < seed_size) {
        temp = rand() % Rule_Max;
        
        for(j=0; j<current_rule_size; j++){
            if (temp == seed[j])
                break;
        }       
        if (j== current_rule_size){
            seed[j] = temp;
            current_rule_size++;
        }
    }
    return seed;
}
long * generate_rules (long * seed, int seed_size, int rule_size, int bit_length){
    long Rule_Max = (long)pow(2,bit_length);
    seed_size = (seed_size < Rule_Max) ? seed_size : Rule_Max;
    long * rule_set = (long *) malloc (sizeof(long) * rule_size);           
    int i;
    for  (i = 0; i < rule_size; i++) {
        rule_set[i] = seed[rand() % seed_size];        
    } 
    return rule_set; 
}

void lb_ub (long *lb, long *ub, int size) {// swap if lower bound > upper bound
    int i;
    long temp;
    for (i =0; i<size; i++) {
        if(lb[i] > ub[i]){
            temp = lb[i];
            lb[i] = ub[i];
            ub[i] = temp;
        }
    }
}

void cout_tree(TreeNode * node){
	if (node == NULL)
		return;
	else
		printf("%lu \n", node->lb);
    
	cout_tree(node->left);
	cout_tree(node->right);		
}
void quickSort(long *array,int left,int right){
    int i=left,j=right;
    long tmp;
    long pivot =array[(left+right)/2];
    while(i<=j){
        while(array[i]<pivot)
            i++;
        while(array[j]>pivot)
            j--;
        if(i<=j){
            tmp = array[i];
            array[i] = array[j];
            array[j] = tmp;
            i++; j--;
        }
    }
    if(left<j)
        quickSort(array,left,j);
    if(i<right)
        quickSort(array,i,right);
}
int Get_node_no (int node_no){
	if (node_no == 1)
		return 1;
	int begin = 1;
	while (begin < node_no)
		begin*=2;
	return begin/2-1;
}
int trim_bound (long *array,int left,int right){
	quickSort(array,left,right);
	int j= left;
    int i;
	for (i= left+1; i<right; i++){
		if (array[i]>array[j]) {
      j++;
      array[j] = array[i];
    }			
	}
	return Get_node_no(j);;
}
void init_points(Point * points, long * bounds, int num){
    int i,j;
    for (i =0; i < num ; i++){
        points[i].value = bounds[i];
        points[i].set = new Set;
    }
}
TreeNode* RangesToBST(Range* ranges, int start, int end){
	if (start > end) return NULL;
    // same as (start+end)/2, avoids overflow.
    int mid = start + (end - start) / 2; 
    TreeNode * node = (TreeNode*) malloc(sizeof(TreeNode));  
    node->left = RangesToBST(ranges, start, mid-1);
    node->right = RangesToBST(ranges, mid+1, end);
    node->ub = ranges[mid].ub;
    node->lb = ranges[mid].lb;
    node->BV = ranges[mid].BV;
    return node;
}

Point * ConstructTree_points(long *seed , int seed_size, long * rule, int rule_size){
    int i,j;
    quickSort(seed, 0, seed_size-1);
    Point * points = (Point *) malloc (sizeof(Point) * seed_size);
    init_points(points, seed, seed_size);   
    for (i =0 ; i<seed_size ; i++){
        for(j=0; j< rule_size; j++){
            if ( points[i].value == rule[j]){								
               (points[i].set)->size++;
            }    				
        }	
        int * set = new int[(points[i].set)->size];	
        int no = 0;
        (points[i].set)->size = no;
        for(j=0; j< rule_size; j++){
         		 if ( points[i].value == rule[j]){								
               set[no]=j;
               no++;
          	 }     				
        }	
        (points[i].set)->element = set;
        free(set);
    }
    return points;
}

void cout_BV (int * BV, int NUM_RULES){
	int i;
	for (i =0 ; i< NUM_RULES/50; i++)
		printf("%d ", BV[i]);	
	printf("\n");	
}
unsigned int Gen_BV_tree_search (long input, long * TreeArray, int TreeSize){	
	int index = 0;
	while(index<TreeSize){
		if (input < TreeArray[index])
			index = index*2+1;
		else
			index = index*2+2;
	}
	return index-TreeSize;
}
void Generate_BV(void* foo){ 
		thread_param_t * myParams = (thread_param_t*) foo;
		int round=0;
		int index;
		for (round = 0; round < NUM_PACKET; round++) { 
			index = Gen_BV_tree_search (myParams->input[round], myParams->TreeArray, myParams->TreeSize);
			(myParams->output)[round%TRACE] = myParams->Lists[index];
		}  
}

void Lookup_Hash(void* foo){ 
	 thread_param_t_H * myParams = (thread_param_t_H*) foo;
	 int i; 
   for (i=0; i< NUM_PACKET; i++) {    
			if(myParams->Table[myParams->trace[i]%myParams->hash_function_key].key == myParams->trace[i])
				myParams->output[i%TRACE] = *(myParams->Table[myParams->trace[i]%myParams->hash_function_key].set);
			else if(myParams->Table[(myParams->trace[i]/myParams->hash_function_key)%myParams->hash_function_key].key == myParams->trace[i])
				myParams->output[i%TRACE] = *(myParams->Table[(myParams->trace[i]/myParams->hash_function_key)%myParams->hash_function_key].set); 		
   		else
   			myParams->output[i%TRACE].element = NULL;
   }
}

void travese_set (Set a){
	int i=0;
	int index;
	while(i<a.size){
		index = a.element[i];
		i++;
	}
}

void Merge(void* foo){ 
	merge_thread_param_t * myParams = (merge_thread_param_t*) foo;
	int i,j;
	for (i=0; i< NUM_PACKET; i++) {
			travese_set(myParams->Set1[i%TRACE]);	
			travese_set(myParams->Set2[i%TRACE]);
			travese_set(myParams->Set3[i%TRACE]);
			travese_set(myParams->Set4[i%TRACE]);
			travese_set(myParams->Set5[i%TRACE]);
			travese_set(myParams->Set6[i%TRACE]);
			travese_set(myParams->Set7[i%TRACE]);
			travese_set(myParams->Set8[i%TRACE]);
			travese_set(myParams->Set9[i%TRACE]);
	}	
}  

long find_empty_slot (Point * points, int * point_index, char method, hash_slot * cuckoo, int function_key, char * function, int * index, int * kick_out_method){
   if (method == 1) {
        if(cuckoo[points[*point_index].value%function_key].key == -1){ // use hash function 1 find an empty slot
            cuckoo[points[*point_index].value%function_key].key = points[*point_index].value;
            cuckoo[points[*point_index].value%function_key].set = points[*point_index].set;
            function[points[*point_index].value%function_key] = 1;                    
            index[points[*point_index].value%function_key] = *point_index;
            return 0;
        }
        else {
            * kick_out_method = function[points[* point_index].value%function_key];
            int temp = * point_index;
            *point_index = index[points[*point_index].value%function_key];            
            cuckoo[points[temp].value%function_key].key = points[temp].value;
            cuckoo[points[temp].value%function_key].set = points[temp].set;
            index[points[temp].value%function_key] = temp;
            function[points[temp].value%function_key] = 1; 
            return points[* point_index].value;
        }
    }
    else if (method == 2){
        if(cuckoo[(points[*point_index].value/function_key)%function_key].key == -1){
            cuckoo[(points[*point_index].value/function_key)%function_key].key = points[*point_index].value;
            cuckoo[(points[*point_index].value/function_key)%function_key].set = points[*point_index].set;
            function[(points[*point_index].value/function_key)%function_key] = 2;                    
            index[(points[*point_index].value/function_key)%function_key] = *point_index;
            return 0;
        }
        else {
            * kick_out_method = function[(points[*point_index].value/function_key)%function_key];
            int temp = * point_index;
            *point_index = index[(points[*point_index].value/function_key)%function_key];    
            cuckoo[(points[temp].value/function_key)%function_key].key = points[temp].value;
            cuckoo[(points[temp].value/function_key)%function_key].set = points[temp].set;
            index[(points[temp].value/function_key)%function_key] = temp;
            function[(points[temp].value/function_key)%function_key] = 2; 
            return points[* point_index].value;
        }       
    }
}

int generate_hash_table (hash_slot * cuckoo, int hash_size, Point * points, int point_size, int function_key){
    if (function_key == 0){
        cout<<"please increase table size";
        return 2;
    }    
    int i,j,k;
    char function[hash_size];
    int index[hash_size];
    for (i=0; i<hash_size; i++){
        function[i] = 0;
        index[i] = 0;
    }
    for (i=0; i<point_size; i++) {
        long kick_out = points[i].value;
        int kick_out_method = 2; // initial use function 1
        int input_index = i;
        while(kick_out != 0){ 
            if(kick_out_method == 2) // use hash 1
                kick_out = find_empty_slot(points, &input_index, 1, cuckoo, function_key, function, index, &kick_out_method);
            else    // use hash 2          
                kick_out = find_empty_slot(points, &input_index, 2, cuckoo, function_key, function, index, &kick_out_method);
            
            if(kick_out == points[i].value) {   
                return 0; // failure
            }
        }
    }    
    return 1; //success
}
void Gen_tree_2 (long *array, int index, int array_size, long * bounds, int start, int end){
	 if (index > array_size) return;
	 if (start == end) { array[index] = bounds[start]; return;} 
	 int mid = start + (end - start) / 2;
	 array[index] = bounds[mid];
	 Gen_tree_2 (array,index*2+1,array_size, bounds, start, mid-1);
	 Gen_tree_2 (array,index*2+2,array_size, bounds, mid+1, end);
}