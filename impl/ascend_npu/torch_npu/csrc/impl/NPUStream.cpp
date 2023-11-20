#include <array>
#include <climits>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>
#include <sys/time.h>
#include <unistd.h>
#include <iostream>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
//#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUQueue.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "acl/acl_rt.h"

namespace c10_npu {

 aclrtStream NPUStream::stream() const {

}

NPUStream getNPUStreamFromPool(c10::DeviceIndex device_index) {

}

NPUStream getStreamFromPool(
    const bool isHighPriority,
    c10::DeviceIndex device_index) {

}

NPUStream getDefaultNPUStream(c10::DeviceIndex device_index) {

}

NPUStream getCurrentNPUStream(c10::DeviceIndex device_index) {

}

NPUStream getCurrentSecondaryStream(c10::DeviceIndex device_index) {

}

aclrtStream getCurrentNPUStreamNoWait(c10::DeviceIndex device_index) {

}

NPUStatus emptyAllNPUStream() {

  return SUCCESS;
}

bool npuSynchronizeDevice(bool check_error) {

}

void enCurrentNPUStream(
    void* cur_paras,
    c10::DeviceIndex device_index) {

}

void setCurrentNPUStream(NPUStream stream) {

}

std::ostream& operator<<(std::ostream& stream, const NPUStream& s) {
  return stream << s.unwrap();
}

void NPUStream::setDataPreprocessStream(bool is_data_preprocess_stream) {

}

bool NPUStream::isDataPreprocessStream() {

}

aclrtStream NPUStream::stream(const bool need_empty) const {

}

} // namespace c10_npu
