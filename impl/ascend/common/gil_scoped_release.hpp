#pragma once

#include "Python.h"

namespace diopi {

// https://docs.python.org/zh-cn/3/c-api/init.html?highlight=pygilstate_check

class gil_scoped_release {
private:
    PyThreadState* state_ = nullptr;

public:
    gil_scoped_release() {
        if (PyGILState_Check()) {
            state_ = PyEval_SaveThread();
        }
    }

    ~gil_scoped_release() {
        if (state_ != nullptr) {
            PyEval_RestoreThread(state_);
            state_ = nullptr;
        }
    }
};

}  // namespace diopi
