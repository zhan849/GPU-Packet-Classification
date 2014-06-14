#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cublas.h>

unsigned int power(unsigned int a, int b){
	unsigned int result = 1;
	int i;
	for(i=0; i<b; i++)
		result= result*a;
	return result;
}	

int main(int argc, char ** argv){
		FILE *fp;
		int i, j, k;
		unsigned int Tree1[15], Tree2[15], BV1[15], BV2[15];
		unsigned int field1[4096], field2[4096]; // used to store the packet field
		
		
		Tree1[0]=1462; 	BV1[0]= power(2,0)+power(2,1); 									Tree2[0]=80;    BV2[0]= power(2,16)+power(2,21);
		Tree1[1]=967; 	BV1[1]= power(2,2)+power(2,6);									Tree2[1]=41;    BV2[1]= power(2,17)+power(2,28)+power(2,4);
		Tree1[2]=1563; 	BV1[2]= power(2,16)+power(2,21);								Tree2[2]=121;   BV2[2]= power(2,13);
		Tree1[3]=387; 	BV1[3]= power(2,15)+power(2,31);								Tree2[3]=23;    BV2[3]= power(2,19)+power(2,12);
		Tree1[4]=1042; 	BV1[4]= power(2,10)+power(2,18)+power(2,22);		Tree2[4]=52;    BV2[4]= power(2,27); 
		Tree1[5]=1477; 	BV1[5]= power(2,7)+power(2,8)+power(2,9);				Tree2[5]=94;    BV2[5]= power(2,3)+power(2,18)+power(2,22);	; 
		Tree1[6]=1870; 	BV1[6]= power(2,17)+power(2,28)+power(2,4);			Tree2[6]=130;   BV2[6]= power(2,10);
		Tree1[7]=12; 		BV1[7]= power(2,3);															Tree2[7]=12;    BV2[7]= power(2,7)+power(2,8)+power(2,9);
		Tree1[8]=497; 	BV1[8]= power(2,11)+power(2,5)+power(2,14);			Tree2[8]=25;    BV2[8]= power(2,11)+power(2,14);
		Tree1[9]=1011; 	BV1[9]= power(2,12);														Tree2[9]=45;    BV2[9]= power(2,15)+power(2,20); 
		Tree1[10]=1300; BV1[10]= power(2,13);														Tree2[10]=55;   BV2[10]= power(2,22)+power(2,23)+power(2,24);
		Tree1[11]=1465; BV1[11]= power(2,26)+power(2,25)+power(2,24);		Tree2[11]=85;   BV2[11]= power(2,31)+power(2,26);
		Tree1[12]=1500; BV1[12]= power(2,29);														Tree2[12]=95;   BV2[12]= power(2,30)+power(2,25);
		Tree1[13]=1600; BV1[13]= power(2,30);														Tree2[13]=126;  BV2[13]= power(2,2)+power(2,29);
		Tree1[14]=1983; BV1[14]= power(2,19)+power(2,27);								Tree2[14]=172;  BV2[14]= power(2,0)+power(2,1)+power(2,5)+power(2,6);
		//printf("%d\n", BV2[14]);

		// read packet data
		if ((fp=fopen("packet.txt","r"))==NULL)
			printf("Cannot open file. Check the name.\n"); 
		else {
			for(i=0;i<4096;i++){        
				fscanf(fp,"%d %d\n",&field1[i], &field2[i]);
	  	}
	  	fclose(fp);
		}
	  	
		// your codes begin
		
		
		// print the results here
		
		
		//
		return 0;
}		