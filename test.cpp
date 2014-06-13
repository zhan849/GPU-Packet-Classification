#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
//#include <cublas.h>
using namespace std;
#define field 15
#define rule 1023
#define packet_num 5

int** tree_gen(int f, int r){
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

int** headers_gen(int num, int f){
	int** headers = new int*[f];
	for (int i = 0; i < f; i++){
		headers[i] = new int[num];
		for (int j = 0; j < num; j++){
			headers[i][j] = rand()%1000;
		}
	}
	return headers;
};
int main(int argc, char** argv){
	int num = atoi(argv[1]);
	int myarray[num];

	//cout<<"start"<<endl;
	srand(time(NULL));
	int tree[field][rule];
	for(int i = 0; i < field; i++){
		tree[i][0] = rand() % 100;
		int temp[rule];
		temp[0] = tree[i][0];
		for (int j = 1; j < rule; j++){
			temp[j] = temp[j-1] + rand() % 20;
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
	for (int i = 0; i < rule; i++){
		cout<<"Rule "<<i<<": ";
		for (int j = 0; j < field; j++){
			cout<<tree[j][i]<<" ";
		}
		cout<<endl;
	}

	int A = 16 & 16;
	int B = 16 | 16;
	int C = 16 & 16 & 16 & 16;
	int D = 16 | 15;
	cout<<"A = "<<A<<", B = "<<B<<", C = "<<C<<", D = "<<D<<endl;

	for (int i = 0; i < num; i++){
		myarray[i] = i;
	}
	for(int i = 0; i < num; i++){
		cout<<myarray[i]<<endl;
	}
	return 0;
}