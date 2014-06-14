Q.1
Unoptimized version:
Here the matrices are brought from the global memory everytime and hence the computation time is more. The total time is 251.549927 ms.

Optimized version:
Here the matrix multiplication is done block by block. So, 1 block each of Matrix A and B are brought from global memory to shared memory. The threads work on these data and then the next block of data is brought from the global memory to shared memory. The main difference is the use of __syncthreads(). This is because first all threads must bring their respective data and synchronize and then all the threads must be done with their part of the computation before the next block of data can be brought from the global memory. Here, the total computation time is 16.612064 ms.

Q.2
In this case, there are 15 threads per block and there are 69 blocks. So, total 1035 threads are used. Each thread will work on 4 packets except the threads in the last few blocks. The number 15 is used because. there are 15 data items to be brought from the global memory to shared memory. So, in 1 cycle all the 15 items are brought. Similarly, the remaining data items are brought and then threads are synchronized. Now, the threads access their data using the index based on block Id and Dim, thread Id and Dim. To make sure, we do not access data beyond 4096 (because 1035 x 4 = 4140), the threads whose data location is beyond 4095 are made to go to sleep. This happens only in the last iteration and for the last few blocks. To search, if there is a match, local variables are used and a binary search is performed. First field is searched and then the search is done for the 2nd field. And the final Bitvector is saved in the global memory. 
