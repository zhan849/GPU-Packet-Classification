#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
//#include <cublas.h>
using namespace std;
#define field 15
#define rule 31
#define packet_num 5

int main(){
	srand(time(NULL));
	bool array1[8], array2[8], array3[8], array4[8];

	for (int i = 0; i < 8; i++){
		if(rand() % 3 == 1){
			array1[i] = true;
		}else{
			array1[i] = false;
		}

		if(rand() % 5 == 2 ){
			array2[i] = true;
		}else{
			array2[i] = false;
		}

		if(rand() % 2 == 0){
			array3[i] = true;
		}else{
			array3[i] = false;
		}
	}
	for (int i = 0; i < 8; i++){
		array4[i] = array1[i] & array2[i] & array3[i];
		cout<<array1[i]<<" "<<array2[i]<<" "<<array3[i]<<" "<<array4[i]<<endl;
	}
	cout<<"size test: "<<sizeof(int)<<" "<<sizeof(bool)*8<<" "<<sizeof(char)<<" "<<sizeof( long int)<<" "<<sizeof(string)<<endl;

	return 0;
}