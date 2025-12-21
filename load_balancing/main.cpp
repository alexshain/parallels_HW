#include <mpi.h>						
#include <iostream>						
#include <queue>						
#include <pthread.h>						
						
struct Job{						
	int jobSimulationTime;					
	Job(int jst):jobSimulationTime(jst){}					
	Job():jobSimulationTime(-1){}					
	process()					
	{					
		sleep(jobSimulationTime);				
	}					
};						
						
std::queue<Job> jobQueue;						
						
bool canBeMoreWork = true;						
pthread_mutex_t mutex;						
pthread_mutex_t mutex_cond;						
pthread_cond_t c_jobAdded, c_needToFetch;						
bool jobAdded, needToFetch;						
						
void* worker(void* args)						
{						
	while(canBeMoreWork)					
	{					
		Job job;				
		pthread_mutex_lock(&mutex);				
		if(!jobQueue.empty())				
		{				
			job = jobQueue.front();			
			jobQueue.pop();			
		} else {				
			pthread_cond_signal(&c_needToFetch);			
		}				
		pthread_mutex_unlock(&mutex);				
		job.process();				
	}					
}						
						
void* fetcher(void* args)						
{						
/*						
	pthread_mutex_lock(&mutex_cond);					
	if(!needToFetch)					
		pthread_cond_wait(&c_needToFetch, &mutex_cond);				
	//проанализировать ситуацию, сохранить статус в локальные переменные					
	pthread_mutex_unlock(&mutex_cond);					
	//обработать ситуацию					
*/						
	int group_size = ((int*)args)[0];					
	int my_rank = ((int*)args)[1];					
	std::vector<bool> finished(group_size, false);					
	int finishedCount = 1; //because we will not request jobs from ourselves					
	int request = -1;					
	int response = -1;					
	while(finshedCount<group_size){					
		for(int r = 0; r<group_size; r++)				
		{				
			if(r != my_rank && !finished[r]){			
				MPI_Send(&request, 1, MPI_INT, r, 0, MPI_COMM_WORLD);		
				MPI_Recv(&response, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);		
				//if got job		
				if(response != -1){		
					Job job(response);	
					pthread_mutex_lock(&mutex);	
					job = jobQueue.push(job);	
					pthread_mutex_unlock(&mutex);	
				}		
				else {		
					finished[r] = true;	
					finishedCount ++;	
				}		
						
			}			
						
		}				
						
	}					
	//another way to stop responder					
	//int mayLeave = -2;					
	//MPI_Send(&mayLeave, 1, MPI_INT, my_rank, 0, MPI_COMM_WORLD);					
	return NULL;					
}						
						
void* responder(void* args)						
{						
	int group_size = ((int*)args)[0];					
	int my_rank = ((int*)args)[1];					
	int rejectedCount = 1;					
						
	while(rejectedCount < group_size)					
	{					
		int request = -100;				
		MPI_Status st;				
		MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, 0, &st);				
//		if(request == -2)				
//		{				
//			someOneCanAsk = false;			
//		} else {				
		Job job;				
		pthread_mutex_lock(&mutex);				
		if(!jobQueue.empty())				
		{				
			job = jobQueue.front();			
			jobQueue.pop();			
			response = job.jobSimulationTime;			
		}				
		else {				
			response = -1;			
			rejectedCount ++;			
		}				
						
		pthread_mutex_unlock(&mutex);				
//		}				
		MPI_Send(&response, 1, MPI_INT, st.MPI_SOURCE, 0, MPI_COMM_WORLD);				
	}					
}						
						
int main(int argc, char** argv)						
{						
	int required = MPI_THREAD_MULTIPLE;					
	int provided = -1;					
	MPI_Init_thread(&argc, &argv, required, &provided);					
	MPI_Comm_size(MPI_COMM_WORLD, &size);					
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);					
						
	if(required != provided)					
	{					
		std::cerr << "Your MPI implementation does not support MPI_THREAD_MULTIPLE.\nCan not proceed." << endl;				
		abort();				
	}					
						
	pthread_t d_worker, d_fetcher, d_responder;					
						
	pthread_attr_t attr;					
	pthread_attr_init(&attr);					
						
	pthread_mutex_init(&mutex);					
	pthread_cond_init(&c_jobAdded);					
	pthread_cond_init(&c_needToFetch);					
						
	pthread_create(&d_worker, &attr, worker, NULL);					
	int args[] = {size, rank};					
	pthread_create(&d_fetcher, &attr, fetcher, args);					
	pthread_create(&d_responder, &attr, responder, args);					
						
	pthread_join(d_worker, nullptr);					
	pthread_join(d_fetcher, nullptr);					
	pthread_join(d_responder, nullptr);					
	std::cout << "All done" << endl;					
	return 0;					
}						
