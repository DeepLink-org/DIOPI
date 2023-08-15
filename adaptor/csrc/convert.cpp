#include "convert.hpp"

std::vector<int64_t> calcStrides(diopiSize_t size, diopiMemoryFormat_t format) {
    size_t ndims = size.len;
    std::vector<int64_t> strides(ndims);
    int64_t st = 1;
    if (format == diopiMemoryFormat_t::Contiguous) {
        for (int64_t i = ndims; i > 0; --i) {
            strides[i - 1] = st;
            if (size.data[i - 1] == 0) continue;
            if (size.data[i - 1] == -1) st = -1;
            if (st != -1) st *= size.data[i - 1];
        }
    } else if (format == diopiMemoryFormat_t::ChannelsLast) {
        for (auto k : {1, 3, 2, 0}) {
            strides[k] = st;
            if (size.data[k] == 0) {
                continue;
            }
            if (size.data[k] == -1) st = -1;
            if (st != -1) st *= size.data[k];
        }
    } else if (format == diopiMemoryFormat_t::ChannelsLast3d) {
        for (auto k : {1, 4, 3, 2, 0}) {
            strides[k] = st;
            if (size.data[k] == 0) {
                continue;
            }
            if (size.data[k] == -1) st = -1;
            if (st != -1) {
                st *= size.data[k];
            }
        }

    } else if (format == diopiMemoryFormat_t::ChannelsLast1d) {
        for (auto k : {1, 2, 0}) {
            strides[k] = st;
            if (size.data[k] == 0) {
                continue;
            }
            if (size.data[k] == -1) st = -1;
            if (st != -1) {
                st *= size.data[k];
            }
        }
    } else {
        // PARROTS_THROW(InvalidArgs) <<
        //         "Invalid MemoryFormat " << memoryFormatName(format);
    }
    return strides;
}
std::vector<diopiMemoryFormat_t> obtainTargetMemoryFormats(size_t shapeLen, std::vector<diopiMemoryFormat_t> supportMemoryFormats) {
    std::vector<diopiMemoryFormat_t> matchedMemoryFormat = matchMemoryFormatBySize(shapeLen);
    supportMemoryFormats = setIntersection(matchedMemoryFormat, supportMemoryFormats);
    return supportMemoryFormats;
}

bool isLikeChannelsLast(diopiConstTensorHandle_t tensor, bool checkContiguous, diopiMemoryFormat_t format) {
    diopiSize_t shape, stride;
    diopiGetTensorShape(tensor, &shape);
    diopiGetTensorStride(tensor, &stride);
    if (shape.len != 4) return false;
    int64_t totalSize = 1;
    for (int64_t i = 0; i < shape.len; ++i) {
        totalSize *= shape.data[i];
    }
    if (totalSize == 0) return false;
    if (stride.data[0] == stride.data[1]) return false;
    if (checkContiguous) {
        auto realStride = calcStrides(shape, format);
        for (int i = 0; i < stride.len; ++i) {
            if (i >= realStride.size() || realStride[i] != stride.data[i]) {
                return false;
            }
        }
        return true;
    } else {
        int64_t st = 1;
        std::vector<int> orders;
        if (format == diopiMemoryFormat_t::ChannelsLast)
            orders = {1, 3, 2, 0};
        else if (format == diopiMemoryFormat_t::ChannelsLast3d)
            orders = {1, 4, 3, 2, 0};
        for (auto k : orders) {
            if (stride.data[k] < st) return false;
            st = stride.data[k] * shape.data[k];
        }
        return true;
    }
}

diopiMemoryFormat_t probableMemoryFormat(diopiConstTensorHandle_t tensor, bool exactMatch) {
    return isLikeChannelsLast(tensor, exactMatch)
               ? diopiMemoryFormat_t::ChannelsLast
               : (isLikeChannelsLast(tensor, exactMatch, diopiMemoryFormat_t::ChannelsLast3d) ? diopiMemoryFormat_t::ChannelsLast3d
                                                                                              : diopiMemoryFormat_t::Contiguous);
}

bool isContiguous(diopiSize_t size, diopiSize_t stride_diopi, diopiMemoryFormat_t format) {
    auto dim = size.len;
    auto shape = size.data;
    auto strides = stride_diopi.data;
    int64_t stride = 1;

    if (format == diopiMemoryFormat_t::Contiguous) {
        for (int64_t i = dim - 1; i >= 0; i--) {
            const auto &shapeD = shape[i];
            if (shapeD != 1) {
                if (strides[i] != stride) {
                    return false;
                }
            }
            stride *= shapeD;
        }
    } else if (format == diopiMemoryFormat_t::ChannelsLast) {
        if (dim != 4) return false;
        for (auto &i : {1, 3, 2, 0}) {
            const auto &shapeD = shape[i];
            if (shapeD != 1) {
                // shape_d != 1 help dealing with shape like [2, 2048, 1, 1]
                if (strides[i] != stride) {
                    return false;
                }
            }
            stride *= shapeD;
        }
    } else if (format == diopiMemoryFormat_t::ChannelsLast3d) {
        if (dim != 5) return false;
        for (auto &i : {1, 4, 3, 2, 0}) {
            const auto &shapeD = shape[i];
            if (shapeD != 1) {
                if (strides[i] != stride) {
                    return false;
                }
            }
            stride *= shape[i];
        }
    } else if (format == diopiMemoryFormat_t::ChannelsLast1d) {
        if (dim != 3) return false;
        for (auto &i : {1, 2, 0}) {
            const auto &shapeD = shape[i];
            if (shapeD != 1) {
                if (strides[i] != stride) {
                    return false;
                }
            }
            stride *= shape[i];
        }
    }
    return true;
}

std::vector<diopiMemoryFormat_t> matchMemoryFormatBySize(size_t sizeLen) {
    std::vector<diopiMemoryFormat_t> matchedMemoryFormat;
    switch (sizeLen) {
        case 1:
        case 2:
            return {diopiMemoryFormat_t::Contiguous};
        case 3:
            return {diopiMemoryFormat_t::Contiguous, diopiMemoryFormat_t::ChannelsLast1d};
        case 4:
            return {diopiMemoryFormat_t::Contiguous, diopiMemoryFormat_t::ChannelsLast};
        case 5:
            return {diopiMemoryFormat_t::Contiguous, diopiMemoryFormat_t::ChannelsLast3d};
        default:
            return {diopiMemoryFormat_t::Contiguous};
    }
}

std::vector<diopiMemoryFormat_t> setIntersection(std::vector<diopiMemoryFormat_t> matchedMemoryFormat, std::vector<diopiMemoryFormat_t> supportMemoryFormat) {
    // A small number of elements are suitable for this method getting
    // intersection
    std::vector<diopiMemoryFormat_t> ret;
    for (auto i : matchedMemoryFormat) {
        if (std::find(supportMemoryFormat.begin(), supportMemoryFormat.end(), i) != std::end(supportMemoryFormat)) {
            ret.push_back(i);
        }
    }
    return ret;
}

#define DIOPI_ERROR_TO_STR(err) \
    case err:                   \
        return #err;

const char *getDiopiErrorStr(diopiError_t err) {
    switch (err) {
        DIOPI_ERROR_TO_STR(diopiErrorOccurred)
        DIOPI_ERROR_TO_STR(diopiNotInited)
        DIOPI_ERROR_TO_STR(diopiNoRegisteredStreamCreateFunction)
        DIOPI_ERROR_TO_STR(diopiNoRegisteredStreamDestoryFunction)
        DIOPI_ERROR_TO_STR(diopiNoRegisteredStreamSyncFunction)
        DIOPI_ERROR_TO_STR(diopiNoRegisteredDeviceMemoryMallocFunction)
        DIOPI_ERROR_TO_STR(diopiNoRegisteredDeviceMemoryFreeFunction)
        DIOPI_ERROR_TO_STR(diopiNoRegisteredDevice2DdeviceMemoryCopyFunction)
        DIOPI_ERROR_TO_STR(diopiNoRegisteredDevice2HostMemoryCopyFunction)
        DIOPI_ERROR_TO_STR(diopiNoRegisteredHost2DeviceMemoryCopyFunction)
        DIOPI_ERROR_TO_STR(diopiNoRegisteredGetLastErrorFunction)
        DIOPI_ERROR_TO_STR(diopi5DNotSupported)
        DIOPI_ERROR_TO_STR(diopiDtypeNotSupported)
        default:
            return "diopiUnexpectedError";
    }
}
#undef DIOPI_ERROR_TO_STR

TimeElapsedRecord TimeElapsed::timeElapsedRecord_("op_time.dat");

std::vector<diopiMemoryFormat_t> defaultFormats{};
