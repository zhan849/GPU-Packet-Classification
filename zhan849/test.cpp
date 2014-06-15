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
cout<<"check 0"<<endl;
	int** tree = new int*[field];
		for(int i = 0; i < rule; i++){
			tree[i] = new int[rule];
		}
	cout<<"check 1"<<endl;
	for(int i = 0; i < field; i++){
		tree[i][0] = rand() % 100;
		int temp[rule];
		temp[0] = tree[i][0];
		for (int j = 1; j < rule; j++){
			temp[j] = temp[j-1] + rand() % 20 + 1;
		}
		cout<<"check 2."<<i<<endl;
		int temp_index = rule-1, tree_index = rule -1, level = log(rule+1) / log(2);
		int step_index = level;
		cout<<"level: "<<level<<endl;
		cout<<"Tree #"<<i<<": "<<endl;
		while (step_index >= 1){
			int step = pow(2, (level - step_index + 1));
			while (temp_index >= 0){
				cout<<"temp_index: "<<temp_index<<", tree_index: "<<tree_index<<", step_index: "<<step_index<<", step: "<<step<<", value: "<<temp[temp_index]<<endl;
				tree[i][tree_index] = temp[temp_index];
				temp_index -= step;
				tree_index--;
			}
			step_index--;
			temp_index = rule - 1 - (pow(2, level - step_index) - 1);
			cout<<"New step_index = "<<step_index<<", New temp_index: "<<temp_index<<endl;
		}
		cout<<"end of tree#"<<i<<endl;
	}

	for(int i = 0; i < rule; i++){
		cout<<"Line: "<<i<<": ";
		for(int j = 0; j < field; j++){
			cout<<tree[i][j]<<" ";
		}
		cout<<endl;
	}

	return 0;
}