#pragma once

#include "Python.h"

namespace diopi {

// https://docs.python.org/zh-cn/3/c-api/init.html?highlight=pygilstate_check

class GilScopedRelease {
private:
    PyThreadState* state_ = nullptr;

public:
    GilScopedRelease() {
        if (PyGILState_Check()) {
            state_ = PyEval_SaveThread();
        }
    }

    ~GilScopedRelease() {
        if (state_ != nullptr) {
            PyEval_RestoreThread(state_);
            state_ = nullptr;
        }
    }
};

}  // namespace diopi
