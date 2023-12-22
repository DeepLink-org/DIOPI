#pragma once

#include "Python.h"

namespace diopi {

// https://docs.python.org/zh-cn/3/c-api/init.html?highlight=pygilstate_check

class GilScopedRelease {
private:
    PyThreadState* state = nullptr;

public:
    GilScopedRelease() {
        if (PyGILState_Check()) {
            state = PyEval_SaveThread();
        }
    }

    ~GilScopedRelease() {
        if (state != nullptr) {
            PyEval_RestoreThread(state);
            state = nullptr;
        }
    }
};

}  // namespace diopi
