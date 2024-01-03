#pragma once

namespace diopi {

class StreamLockGuard {
private:
    void* mutex_ = nullptr;

public:
    explicit StreamLockGuard(void* aclrtStreamHandle);
    ~StreamLockGuard();
};

}  // namespace diopi
