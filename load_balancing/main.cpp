#include <mpi.h>
#include <iostream>
#include <queue>
#include <pthread.h>
#include <vector>
#include <unistd.h>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <sstream>

std::mutex logMutex;

void log_message(int rank, const std::string& msg) {
    std::lock_guard<std::mutex> lock(logMutex);
    std::cout << "[Rank " << rank << "] " << msg << std::endl;
}

struct Job {
    int jobSimulationTime;
    Job(int jst) : jobSimulationTime(jst) {}
    Job() : jobSimulationTime(-1) {}
    void process() {
        if (jobSimulationTime > 0) {
            sleep(jobSimulationTime);
        }
    }
};

// === Shared state ===
std::queue<Job> jobQueue;
pthread_mutex_t queueMutex = PTHREAD_MUTEX_INITIALIZER;

// === НОВОЕ: синхронизация между worker и fetcher ===
pthread_mutex_t fetcherMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t workLowCond = PTHREAD_COND_INITIALIZER;
std::atomic<bool> fetcherActive{false}; // true = ищет работу

std::atomic<bool> stopRequested{false};
std::atomic<bool> globalIdle{false};

int worldSize;
int worldRank;

// === Worker ===
void* worker(void* arg) {
    int jobCount = 0;
    while (!stopRequested.load()) {
        Job job;
        bool hasJob = false;
        pthread_mutex_lock(&queueMutex);
        if (!jobQueue.empty()) {
            job = jobQueue.front();
            jobQueue.pop();
            hasJob = true;
        }
        pthread_mutex_unlock(&queueMutex);

        if (hasJob) {
            jobCount++;
            log_message(worldRank, "Worker: processing job #" + std::to_string(jobCount) + " (sleep=" + std::to_string(job.jobSimulationTime) + ")");

            // Проверяем, не стало ли задач мало
            pthread_mutex_lock(&queueMutex);
            bool needMore = (jobQueue.size() <= 1); // порог: ≤1 задача
            pthread_mutex_unlock(&queueMutex);

            if (needMore && !fetcherActive.load()) {
                pthread_mutex_lock(&fetcherMutex);
                fetcherActive = true;
                pthread_cond_signal(&workLowCond);
                pthread_mutex_unlock(&fetcherMutex);
            }

            job.process();
        } else {
            // Если вообще нет задач — активируем fetcher немедленно
            pthread_mutex_lock(&fetcherMutex);
            if (!fetcherActive.load()) {
                fetcherActive = true;
                pthread_cond_signal(&workLowCond);
            }
            pthread_mutex_unlock(&fetcherMutex);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    log_message(worldRank, "Worker: stopping.");
    return nullptr;
}

// === Responder ===
void* responder(void* arg) {
    log_message(worldRank, "Responder: started.");
    int recvCount = 0;
    while (!globalIdle.load()) {
        int request;
        MPI_Status status;
        int flag;

        MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            int src = status.MPI_SOURCE;
            MPI_Recv(&request, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            recvCount++;

            int response = -1;
            bool hadJob = false;
            pthread_mutex_lock(&queueMutex);
            if (!jobQueue.empty()) {
                Job j = jobQueue.front();
                jobQueue.pop();
                response = j.jobSimulationTime;
                hadJob = true;
            }
            pthread_mutex_unlock(&queueMutex);

            MPI_Send(&response, 1, MPI_INT, src, 0, MPI_COMM_WORLD);
            log_message(worldRank, "Responder: handled request #" + std::to_string(recvCount) + " from rank " + std::to_string(src) + 
                        " → sent " + (hadJob ? "job(" + std::to_string(response) + ")" : "NO_JOB"));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    log_message(worldRank, "Responder: stopping.");
    return nullptr;
}

// === Fetcher (теперь ленивый) ===
void* fetcher(void* arg) {
    log_message(worldRank, "Fetcher: started (lazy mode).");
    int allgatherCount = 0;

    while (!globalIdle.load()) {
        // Ждём сигнала от worker
        pthread_mutex_lock(&fetcherMutex);
        while (!fetcherActive.load() && !globalIdle.load()) {
            pthread_cond_wait(&workLowCond, &fetcherMutex);
        }
        if (globalIdle.load()) {
            pthread_mutex_unlock(&fetcherMutex);
            break;
        }
        pthread_mutex_unlock(&fetcherMutex);

        log_message(worldRank, "Fetcher: activated — seeking work...");

        bool gotWork = false;
        if (worldSize > 1) {
            // Опрашиваем до тех пор, пока не получим работу или не опросим всех
            for (int r = 0; r < worldSize && !gotWork; ++r) {
                if (r == worldRank) continue;
                int request = 0;
                int response = -1;
                MPI_Send(&request, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                MPI_Recv(&response, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (response > 0) {
                    pthread_mutex_lock(&queueMutex);
                    jobQueue.push(Job(response));
                    pthread_mutex_unlock(&queueMutex);
                    gotWork = true;
                    log_message(worldRank, "Fetcher: got job(" + std::to_string(response) + ") from rank " + std::to_string(r));
                } else {
                    log_message(worldRank, "Fetcher: rank " + std::to_string(r) + " has no job");
                }
            }
        }

        // После попытки — деактивируем себя
        fetcherActive = false;

        // Всё равно делаем Allgather, чтобы проверить глобальное завершение
        int localIdle = (jobQueue.empty()) ? 1 : 0;
        std::vector<int> allIdle(worldSize);
        MPI_Allgather(&localIdle, 1, MPI_INT, allIdle.data(), 1, MPI_INT, MPI_COMM_WORLD);
        allgatherCount++;

        bool nowIdle = std::all_of(allIdle.begin(), allIdle.end(), [](int x) { return x == 1; });
        log_message(worldRank, "Allgather #" + std::to_string(allgatherCount) + ": localIdle=" + std::to_string(localIdle) + 
                    ", globalIdle=" + (nowIdle ? "YES" : "NO"));

        if (nowIdle) {
            globalIdle.store(true);
            stopRequested.store(true);
            log_message(worldRank, "Fetcher: global idle detected → signaling stop.");
            break;
        }
    }
    log_message(worldRank, "Fetcher: stopping.");
    return nullptr;
}

// === Main ===
int main(int argc, char** argv) {
    int required = MPI_THREAD_MULTIPLE;
    int provided;
    MPI_Init_thread(&argc, &argv, required, &provided);

    double t0 = MPI_Wtime();

    if (provided < required) {
        std::cerr << "MPI_THREAD_MULTIPLE not supported!\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    // Initial work
    if (worldRank == 0) {
        for (int i = 0; i < 5; ++i) {
            pthread_mutex_lock(&queueMutex);
            jobQueue.push(Job(2));
            pthread_mutex_unlock(&queueMutex);
        }
    }

    log_message(worldRank, "Starting threads...");

    // === Инициализация новых примитивов ===
    pthread_mutex_init(&fetcherMutex, nullptr);
    pthread_cond_init(&workLowCond, nullptr);

    pthread_t t_worker, t_fetcher, t_responder;
    pthread_create(&t_worker, nullptr, worker, nullptr);
    pthread_create(&t_fetcher, nullptr, fetcher, nullptr);
    pthread_create(&t_responder, nullptr, responder, nullptr);

    pthread_join(t_worker, nullptr);
    pthread_join(t_fetcher, nullptr);
    pthread_join(t_responder, nullptr);

    log_message(worldRank, "All threads joined.");

    // === Уничтожение ===
    pthread_mutex_destroy(&fetcherMutex);
    pthread_cond_destroy(&workLowCond);

    double t1 = MPI_Wtime();

    if (worldRank == 0) {
        std::cout << "\n[SUMMARY] All work completed system-wide.\n" << std::endl;
    }

    if (worldRank == 0) std::cout << "Total time: " << (t1 - t0) << " sec\n";

    MPI_Finalize();
    return 0;
}