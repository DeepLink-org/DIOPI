#include <dlfcn.h>

#include <cstdio>
#include <filesystem>
#include <stdexcept>

void* dynLoadFile(const char* diopiRealName) {
    namespace fs = std::filesystem;
    void* handle = dlopen(diopiRealName, RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!handle) {
        Dl_info info;
        if (dladdr(reinterpret_cast<void*>(dynLoadFile), &info) != 0 && info.dli_fname != nullptr) {
            fs::path fpath(info.dli_fname);
            auto diopiInLoader = fpath.parent_path().append(diopiRealName).string();
            printf(
                "diopi dyload fail, seems LD_LIBRARAY_PATH not contains %s, try to load "
                "from loader current dir's %s \n",
                diopiRealName,
                diopiInLoader.c_str());

            handle = dlopen(diopiInLoader.c_str(), RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND);
        }
    }
    if (!handle) {
        fprintf(stderr,
                "! please note that dynamic loaded diopi_impl.so need explictly link to it's \
                 diopi_rt (now is torch_dipu), so it cannot be used for diopi-test now \n");
        fprintf(stderr, "%s \n", dlerror());
        throw std::runtime_error("diopi_init err");
    }
    return handle;
}
