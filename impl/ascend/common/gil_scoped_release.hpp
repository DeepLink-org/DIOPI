#pragma once

#include "Python.h"

namespace diopi {

// https://docs.python.org/zh-cn/3/c-api/init.html?highlight=pygilstate_check

class gil_scoped_release {
private:
    PyThreadState* state = nullptr;

public:
    explicit gil_scoped_release() {
        if (PyGILState_Check()) {
            state = PyEval_SaveThread();
        }
    }

    ~gil_scoped_release() {
        if (state != nullptr) {
            PyEval_RestoreThread(state);
            state = nullptr;
        }
    }
};

}  // namespace diopi
