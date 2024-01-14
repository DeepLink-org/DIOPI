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
            beInitedInDiopi = true;
        }
    }
    ~PyInterpreteInit() {
        if (beInitedInDiopi) {
            int exState = Py_FinalizeEx();
            if (exState == -1) {
                std::cerr << "Py_FinalizeEx failed in " << __FILE__ << ":" << __LINE__ << "in func:" << __FUNCTION__ << std::endl;
            }
        }
    }

private:
    bool beInitedInDiopi = false;
};

class InitDiopi {
public:
    InitDiopi() = default;

private:
    // init the module here.
    PyInterpreteInit pyInit;
};

InitDiopi initDiopi;  // initialize diopi
