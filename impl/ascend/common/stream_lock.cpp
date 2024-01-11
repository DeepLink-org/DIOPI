#include "stream_lock.hpp"

#include <map>
#include <mutex>
#include <thread>

namespace diopi {

using MutexType = std::mutex;

namespace {

MutexType* getLockForStream(void* aclrtStreamHandle) {
    static std::map<void*, MutexType> streamThreadMutexMap;
    return &streamThreadMutexMap[aclrtStreamHandle];
}

}  // namespace

StreamLockGuard::StreamLockGuard(void* aclrtStreamHandle) {
    MutexType* mutexPtr = getLockForStream(aclrtStreamHandle);
    mutex_ = mutexPtr;
    mutexPtr->lock();
}

StreamLockGuard::~StreamLockGuard() {
    MutexType* mutexPtr = static_cast<MutexType*>(mutex_);
    mutexPtr->unlock();
}

}  // namespace diopi
