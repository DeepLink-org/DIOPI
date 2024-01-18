/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <Python.h>

#include <iostream>

class PyInterpreteInit {
public:
    PyInterpreteInit() {
        if (!Py_IsInitialized()) {
            Py_Initialize();
            beInitedInDiopi_ = true;
        }
    }
    ~PyInterpreteInit() {
        if (beInitedInDiopi_) {
            int exState = Py_FinalizeEx();
            if (exState == -1) {
                std::cerr << "Py_FinalizeEx failed in " << __FILE__ << ":" << __LINE__ << "in func:" << __FUNCTION__ << std::endl;
            }
        }
    }

private:
    bool beInitedInDiopi_ = false;
};

class InitDiopi {
public:
    InitDiopi() = default;

private:
    // init the module here.
    PyInterpreteInit pyInit_;
};

InitDiopi initDiopi;  // initialize diopi
