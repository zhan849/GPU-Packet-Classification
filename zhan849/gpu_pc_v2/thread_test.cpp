#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
using namespace std;

void *print_message_function( void *ptr );

int main()
{
     pthread_t thread1, thread2;
     const char *message1 = "Thread 1";
     const char *message2 = "Thread 2";
     int  iret1, iret2;

    /* Create independent threads each of which will execute function */

     iret1 = pthread_create( &thread1, NULL, print_message_function, (void*) message1);
     if(iret1)
     {
         fprintf(stderr,"Error - pthread_create() return code: %d\n",iret1);
         exit(EXIT_FAILURE);
     }

     iret2 = pthread_create( &thread2, NULL, print_message_function, (void*) message2);
     if(iret2)
     {
         fprintf(stderr,"Error - pthread_create() return code: %d\n",iret2);
         exit(EXIT_FAILURE);
     }

     printf("pthread_create() for thread 1 returns: %d\n",iret1);
     printf("pthread_create() for thread 2 returns: %d\n",iret2);

     /* Wait till threads are complete before main continues. Unless we  */
     /* wait we run the risk of executing an exit which will terminate   */
     /* the process and all threads before the threads have completed.   */

     pthread_join( thread1, NULL);
     pthread_join( thread2, NULL); 

     iret1 = pthread_create(thread1, NULL, pthread_threadid_np, (void*) NULL);
     iret2 = pthread_create(thread2, NULL, pthread_threadid_np, (void*) NULL);

     exit(EXIT_SUCCESS);
     return 0;
}

void *print_message_function( void *ptr )
{
     char *message1;
     message1 = (char *) ptr;
     printf("%s \n", message1);
}