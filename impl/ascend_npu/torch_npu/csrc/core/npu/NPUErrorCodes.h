#pragma once

#include <unordered_map>

namespace c10_npu::acl {

class AclErrorCode {
public:
    std::unordered_map<int, std::string> error_code_map = {
        {100000, "Parameter verification failed.\nCheck whether the input parameter value of the interface is correct."},
        {100001,
         "ACL uninitialized.\n(1)Check whether the acl.init interface has been invoked for initialization. Ensure that the acl.init interface has been invoked "
         "before other pyACL interfaces.\n(2)Check whether the initialization interface of the corresponding function has been invoked, for example, the "
         "acl.mdl.init_dump interface for initializing Dump and the acl.prof.init interface for initializing Profiling."},
        {100002,
         "Repeated initialization or repeated loading.\nCheck whether the corresponding interface is repeatedly invoked for initialization or loading."},
        {100003, "Invalid file.\nCheck whether the file exists and can be accessed."},
        {100004, "Failed to write the file.\nCheck whether the file path exists and whether the file has the write permission."},
        {100005, "Invalid file size.\nCheck whether the file size meets the interface requirements."},
        {100006, "Failed to parse the file.\nCheck whether the file content is valid."},
        {100007, "The file is missing parameters.\nCheck whether the file content is complete."},
        {100008, "Invalid file parameter.\nCheck whether the parameter values in the file are correct."},
        {100009,
         "Invalid dump configuration.\nCheck whether the dump configuration in the acl.init interface configuration file is correct. For details, see "
         "'Preparing Comparison Data > Preparing Offline Model Dump Data Files' in the <Precision Comparison Tool User Guide>."},
        {100010, "Invalid Profiling configuration.\nCheck whether the profiling configuration is correct."},
        {100011, "Invalid model ID.\nCheck whether the model ID is correct and whether the model is correctly loaded."},
        {100012,
         "Failed to deserialize the model.\nThe model may not match the current version. Convert the model again by referring to the <ATC Tool User Guide>."},
        {100013, "Failed to parse the model.\nThe model may not match the current version. Convert the model again by referring to the <ATC Tool User Guide>."},
        {100014, "Failed to read the model.\nCheck whether the model file exists and can be accessed."},
        {100015, "Invalid model size.\nThe model file is invalid, Convert the model again by referring to the <ATC Tool User Guide>."},
        {100016,
         "The model is missing parameters.\nThe model may not match the current version. Convert the model again by referring to the <ATC Tool User Guide>."},
        {100017, "The input for the model does not match.\nCheck whether the model input is correct."},
        {100018, "The output of the model does not match.\nCheck whether the output of the model is correct."},
        {100019,
         "Model is not-dynamic.\nCheck whether the current model supports dynamic scenarios. If not, convert the model again by referring to the <ATC Tool "
         "User Guide>."},
        {100020, "The type of a single operator does not match.\nCheck whether the operator type is correct."},
        {100021, "The input of a single operator does not match.\nCheck whether the operator input is correct."},
        {100022, "The output of a single operator does not match.\nCheck whether the operator output is correct."},
        {100023, "The attributes of a single operator do not match.\nCheck whether the operator attributes are correct."},
        {100024, "Single operator not found.\nCheck whether the operator type is supported."},
        {100025,
         "Failed to load a single operator.\nThe model may not match the current version. Convert the model again by referring to the <ATC Tool User Guide>."},
        {100026, "Unsupported data type.\nCheck whether the data type exists or is currently supported."},
        {100027, "The format does not match.\nCheck whether the format is correct."},
        {100028,
         "When the operator interface is compiled in binary selection mode, the operator haven't registered a selector.\nCheck whether the "
         "acl.op.register_compile_func interface is invoked to register the operator selector."},
        {100029,
         "During operator compilation, the operator kernel haven't registered.\nCheck whether the acl.op.create_kernel interface is invoked to register the "
         "operator kernel."},
        {100030,
         "When the operator interface is compiled in binary selection mode, the operator is registered repeatedly.\nCheck whether the "
         "acl.op.register_compile_func interface is repeatedly invoked to register the operator selector."},
        {100031,
         "During operator compilation, the operator kernel is registered repeatedly.\nCheck whether the acl.op.create_kernel interface is repeatedly invoked "
         "to register the operator kernel."},
        {100032, "Invalid queue ID.\nCheck whether the queue ID is correct."},
        {100033, "Duplicate subscription.\nCheck whether the acl.rt.subscribe_report interface is invoked repeatedly for the same stream."},
        {100034,
         "The stream is not subscribed.\nCheck whether the acl.rt.subscribe_report interface has been invoked.\n[notice] This return code will be discarded in "
         "later versions. Use the 'ACL_ERROR_RT_STREAM_NO_CB_REG' return code."},
        {100035,
         "The thread is not subscribed.\nCheck whether the acl.rt.subscribe_report interface has been invoked.\n[notice] This return code will be discarded in "
         "later versions. Use the 'ACL_ERROR_RT_THREAD_SUBSCRIBE' return code."},
        {100036,
         "Waiting for callback time out.\n(1)Check whether the acl.rt.launch_callback interface has been invoked to deliver the callback task.\n(2)Check "
         "whether the timeout interval in the acl.rt.process_report interface is proper.\n(3)Check whether the callback task has been processed. If the "
         "callback task has been processed but the acl.rt.process_report interface is still invoked, optimize the code logic.\n[notice] This return code will "
         "be discarded in later versions.Use the 'ACL_ERROR_RT_REPORT_TIMEOUT' return code."},
        {100037, "Repeated deinitialization.\nCheck whether the acl.finalize interface is repeatedly invoked for deinitialization."},
        {100038,
         "The static AIPP configuration information does not exist.\nWhen invoking the 'acl.mdl.get_first_aipp_info' interface, ensure that the index "
         "value is "
         "correct.\n[notice] This return code will be discarded in later versions. Use the 'ACL_ERROR_GE_AIPP_NOT_EXIST' return code."},
        {100039,
         "The dynamic library path configured before running the application is the path of the compilation stub, not the correct dynamic library "
         "path.\nCheck "
         "the configuration of the dynamic library path and ensure that the dynamic library in running mode is used."},
        {100040, "Group is not set. Check if the aclrtSetGroup interface has been called."},
        {100041,
         "No corresponding Group is created. Check whether the Group ID set during the interface invocation is within the supported range. The value range of "
         "the Group ID is [0, (number of groups -1)]. You can invoke the aclrtGetGroupCount interface to obtain the number of groups."},
        {100042,
         "A profiling data collection task exists. Check whether multiple profiling configurations are delivered to the same device.\nFor details, see the "
         "Profiling pyACL API. (Performance data is collected and flushed to disks through the Profiling pyACL API.) The code logic is adjusted based on the "
         "interface invoking requirements and interface invoking sequence in."},
        {100043,
         "The acl.prof.init interface is not used for analysis initialization. Check the interface invoking sequence for collecting and analyzing data and "
         "analyze the pyACL API by referring to. See (Performance data collected and flushed to disks by analyzing the pyACL API) Description in."},
        {100044,
         "A task for obtaining dump data exists. Before invoking the 'acl.mdl.init_dump', 'acl.mdl.set_dump', 'acl.mdl.finalize_dump' interfaces to configure "
         "dump information: Check whether the acl.init interface has been invoked to configure the dump information. If yes, adjust the code logic and retain "
         "one method to configure the dump information."},
        {100045,
         "The acl.mdl.init_dump interface is not used to initialize the dump. Check the invoking sequence of the interfaces for obtaining dump data. For "
         "details, see the description of the acl.mdl.init_dump interface."},
        {148046,
         "Subscribe to the same model repeatedly. Check the API invoking sequence. See (Performance data collected and flushed to disks by analyzing the pyACL "
         "API) Description in."},
        {148047,
         "A conflict occurs in invoking the interface for collecting performance data. The interfaces for collecting profiling performance data in the two "
         "modes cannot be invoked crossly. The 'acl.prof.model_subscribe', 'acl.prof.get_op_*', 'acl.prof.model_un_subscribe' interface cannot be invoked "
         "between the 'acl.prof.init' and 'acl.prof.finalize' interfaces. The 'acl.prof.init', 'acl.prof.start', 'acl.prof.stop', and 'acl.prof.finalize' "
         "interfaces cannot be invoked between the 'acl.prof.model_subscribe' and 'acl.prof.model_un_subscribe' interfaces."},
        {148048,
         "Invalid operator cache information aging configuration. Check the aging configuration of the operator cache information. For details, see the "
         "configuration description and example in acl.init."},
        {148049,
         "The 'ASCEND_OPP_PATH' environment variable is not set, or the value of the environment variable is incorrect. Check whether the 'ASCEND_OPP_PATH' "
         "environment variable is set and whether the value of the environment variable is the installation path of the opp software package."},
        {148050,
         "The operator does not support dynamic Shape. Check whether the shape of the operator in the single-operator model file is dynamic. If yes, change "
         "the shape to a fixed one. Check whether the shape of aclTensorDesc is dynamic during operator compilation. If yes, create aclTensorDesc based on the "
         "fixed shape."},
        {148051,
         "Related resources have not been released. This error code is returned if the related channel is not destroyed when the channel description "
         "information is being destroyed. Check whether the channel associated with the channel description is destroyed."},
        {148052,
         "Input image encoding format (such as arithmetic encoding and progressive encoding) that is not supported by the JPEGD function. When JPEGD image "
         "decoding is implemented, only Huffman encoding is supported. The color space of the original image before compression is YUV. The ratio of pixel "
         "components is  4:4:4, 4:2:2, 4:2:0, 4:0:0, or 4:4:0. Arithmetic coding, progressive JPEG, and JPEG2000 are not supported."},
        {200000, "Failed to apply for memory. Check the available memory in the hardware environment."},
        {200001, "The interface does not support this function. Check whether the invoked interface is supported."},
        {200002,
         "Invalid device. Check whether the device exists. [notice] This return code will be discarded in later versions. Use the "
         "'ACL_ERROR_RT_INVALID_DEVICEID' return code."},
        {200003, "The memory address is not aligned. Check whether the memory address meets the interface requirements."},
        {200004, "The resources do not match. Check whether the correct resources such as Stream and Context are transferred when the interface is invoked."},
        {200005,
         "Invalid resource handle. Check whether the transferred resources such as Stream and Context are destroyed or occupied when the interface is "
         "invoked."},
        {200006,
         "This feature is not supported. Rectify the fault based on the error information in the ascend log or contact Huawei technical support. For details "
         "about logs, see the Log Reference."},
        {200007,
         "Unsupported profiling configurations are delivered. Check whether the profiling configuration is correct by referring to the description in "
         "'acl.prof.create_config'."},
        {300000, "The storage limit is exceeded. Check the remaining storage space in the hardware environment."},
        {500000, "Unknown internal error. Rectify the fault based on the error information in the ascend log."},
        {500001, "The internal ACL of the system is incorrect. Rectify the fault based on the error information in the ascend log."},
        {500002, "A GE error occurs in the system. Rectify the fault based on the error information in the ascend log."},
        {500003, "The internal RUNTIME of the system is incorrect. Rectify the fault based on the error information in the ascend log."},
        {500004, "An internal DRV (Driver) error occurs. Rectify the fault based on the error information in the ascend log."},
        {500005, "Profiling error. Rectify the fault based on the error information in the ascend log."},
        /* following return codes is for the internal RUNTIME */
        {107000, "Parameter verification failed. Check whether the input parameters of the interface are correct."},
        {107001, "Invalid device ID. Check whether the device ID is valid."},
        {107002, "The context is empty. Check whether acl.rt.set_context or acl.rt.set_device is called."},
        {107003, "The stream is not in the current context. Check whether the context where the stream is located is the same as the current context."},
        {107004, "The model is not in the current context. Check whether the loaded model is consistent with the current context."},
        {107005, "The stream is not in the current model. Check whether the stream has been bound to the model."},
        {107006, "The event timestamp is invalid. Check whether the event is created."},
        {107007, "Invert the event timestamp. Check whether the event is created."},
        {107008,
         "The memory address is not aligned.\nCheck whether the applied memory addresses are aligned. For details about the restrictions on the memory "
         "application interface, see Memory Management."},
        {107009, "Failed to open the file.\nCheck whether the file exists."},
        {107010, "Failed to write the file. Check whether the file exists or has the write permission."},
        {107011, "The stream is not subscribed to or subscribed to repeatedly. Check whether the current stream is subscribed to or repeatedly subscribed to."},
        {107012,
         "The thread is not subscribed or subscribed repeatedly. Check whether the current thread subscribes to or subscribes to the thread repeatedly."},
        {107013, "The group is not set."},
        {107014,
         "The corresponding group is not created. Check whether the group ID set when the interface is invoked is within the supported range. The value range "
         "of the group ID is [0, (Number of groups - 1)]."},
        {107015,
         "The stream corresponding to the callback is not registered with the thread. Check whether the stream has been registered with the thread and whether "
         "the acl.rt.subscribe_report interface is invoked."},
        {107016, "Invalid memory type. Check whether the memory type is valid."},
        {107017, "Invalid resource handle. Check whether the input and used parameters are correct."},
        {107018, "The memory type applied for is incorrect. Check whether the input and used memory types are correct."},
        {107019, "Task execution timed out. Re-execute the interface for delivering the task."},
        {207000, "This feature is not supported. Rectify the fault based on the error information in the ascend log."},
        {207001, "Failed to apply for memory. Check the remaining storage space in the hardware environment."},
        {207002, "Failed to release the memory. Rectify the fault based on the error information in the ascend log."},
        {207003, "The operation of the aicore operator overflows. Check whether the corresponding aicore operator operation overflows."},
        {207004, "The device is unavailable. Check whether the device is running properly."},
        {207005, "Failed to apply for memory. Check the remaining storage space in the hardware environment."},
        {207006, "You do not have the operation permission. Check whether the permission of the user who runs the application is correct."},
        {207007,
         "Event resources are insufficient. Check whether the number of events meets the requirements by referring to the description of the "
         "acl.rt.create_event interface."},
        {207008,
         "Stream resources are insufficient. Check whether the number of streams meets the requirements by referring to the description of the "
         "acl.rt.create_stream interface."},
        {207009,
         "Notify resources in the system are insufficient. There are too many concurrent data preprocessing tasks or model inference consumes too many "
         "resources. You are advised to reduce the number of concurrent tasks or uninstall some models."},
        {207010, "Insufficient model resources. You are advised to uninstall some models."},
        {207011, "Runtime internal resources are insufficient. Rectify the fault based on the error information in the ascend log."},
        {207012, "The number of queues exceeds the upper limit. Destroy unnecessary queues before creating new queues."},
        {207013, "The queue is empty. Cannot obtain data from an empty queue. Add data to the queue and then obtain data."},
        {207014, "The queue is full.  You cannot add data to a queue that is full. Obtain data from the queue and then add data."},
        {207015, "The queue is initialized repeatedly.  You are advised to initialize the queue only once."},
        {207018,
         "The memory on the device is exhausted.  Check the memory usage on the device and properly plan the memory usage based on the memory specifications "
         "on the device."},
        {507000, "An internal error occurs in the runtime module on the host.  Rectify the fault based on the error information in the ascend log."},
        {507001, "An internal error occurs in the task scheduler module on the device.  Rectify the fault based on the error information in the ascend log."},
        {507002, "The number of tasks on the stream reaches the maximum.  Rectify the fault based on the error information in the ascend log."},
        {507003, "The number of tasks on the stream is empty.  Rectify the fault based on the error information in the ascend log."},
        {507004, "Not all tasks on the stream are executed.  Rectify the fault based on the error information in the ascend log."},
        {507005, "Task execution on the AI CPU is complete.  Rectify the fault based on the error information in the ascend log."},
        {507006, "The event is not complete.  Rectify the fault based on the error information in the ascend log."},
        {507007, "Failed to release the context.  Rectify the fault based on the error information in the ascend log."},
        {507008, "Failed to obtain the SOC version.  Rectify the fault based on the error information in the ascend log."},
        {507009, "The task type is not supported.  Rectify the fault based on the error information in the ascend log."},
        {507010, "The task scheduler loses the heartbeat.  Rectify the fault based on the error information in the ascend log."},
        {507011, "Model execution failed.  Rectify the fault based on the error information in the ascend log."},
        {507012, "Failed to obtain the task scheduler message.  Rectify the fault based on the error information in the ascend log."},
        {507013, "System Direct Memory Access (DMA) hardware execution error.  Rectify the fault based on the error information in the ascend log."},
        {507014, "The aicore execution times out.  Rectify the fault based on the error information in the ascend log."},
        {507015, "The aicore execution is abnormal.  Rectify the fault based on the error information in the ascend log."},
        {507016, "An exception occurs when the aicore trap is executed.  Rectify the fault based on the error information in the ascend log."},
        {507017, "The aicpu execution times out.  Rectify the fault based on the error information in the ascend log."},
        {507018, "The aicpu execution is abnormal.  Rectify the fault based on the error information in the ascend log."},
        {507019,
         "The AICPU does not send a response to the task scheduler after data dump.  Rectify the fault based on the error information in the ascend log."},
        {507020,
         "The AIPPU does not send a response to the task scheduler after executing the model.  Rectify the fault based on the error information in the ascend "
         "log."},
        {507021, "The profiling function is abnormal.  Rectify the fault based on the error information in the ascend log."},
        {507022, "The communication between processes is abnormal.  Rectify the fault based on the error information in the ascend log."},
        {507023, "The model exits.  Rectify the fault based on the error information in the ascend log."},
        {507024, "The operator is being deregistered.  Rectify the fault based on the error information in the ascend log."},
        {507025, "The ring buffer function is not initialized.  Rectify the fault based on the error information in the ascend log."},
        {507026, "The ring buffer has no data.  Rectify the fault based on the error information in the ascend log."},
        {507027, "The kernel in RUNTIME is not registered.  Rectify the fault based on the error information in the ascend log."},
        {507028, "Repeatedly register the kernel inside the RUNTIME.  Rectify the fault based on the error information in the ascend log."},
        {507029, "The debug function failed to be registered.  Rectify the fault based on the error information in the ascend log."},
        {507030, "Deregistration of the debugging function fails.  Rectify the fault based on the error information in the ascend log."},
        {507031, "The tag is not in the current context.  Rectify the fault based on the error information in the ascend log."},
        {507032, "The number of registered programs exceeds the upper limit.  Rectify the fault based on the error information in the ascend log."},
        {507033, "Failed to start the device.  Rectify the fault based on the error information in the ascend log."},
        {507034, "Vector core execution timed out.  Rectify the fault based on the error information in the ascend log."},
        {507035, "The vector core execution is abnormal.  Rectify the fault based on the error information in the ascend log."},
        {507036, "An exception occurs when vector core traps are executed.  Rectify the fault based on the error information in the ascend log."},
        {507037,
         "An exception occurred when applying for internal resources of the Runtime.  Rectify the fault based on the error information in the ascend log."},
        {507038,
         "An error occurred when modifying the die mode, can not change the die mode.  Rectify the fault based on the error information in the ascend log."},
        {507039, "The die cannot be specified in single-die mode.  Rectify the fault based on the error information in the ascend log."},
        {507040, "The specified die ID is incorrect.  Rectify the fault based on the error information in the ascend log."},
        {507041, "The die mode is not set.  Rectify the fault based on the error information in the ascend log."},
        {507042, "The aicore trap read out-of-bounds exception.  Rectify the fault based on the error information in the ascend log."},
        {507043, "The aicore trap write out-of-bounds exception.  Rectify the fault based on the error information in the ascend log."},
        {507044, "Vector core trap read out-of-bounds exception.  Rectify the fault based on the error information in the ascend log."},
        {507045, "Vector core trap write out-of-bounds exception.  Rectify the fault based on the error information in the ascend log."},
        {507046,
         "In the specified timeout waiting event, all tasks in the specified stream are not completed.  Rectify the fault based on the error information in "
         "the ascend log."},
        {507047,
         "During the specified event synchronization waiting time, the event is not executed completely.  Rectify the fault based on the error information in "
         "the ascend log."},
        {507048, "The execution of the internal task times out.  Rectify the fault based on the error information in the ascend log."},
        {507049, "An exception occurs during the execution of an internal task.  Rectify the fault based on the error information in the ascend log."},
        {507050, "The trap of the internal task is abnormal.  Rectify the fault based on the error information in the ascend log."},
        {507051, "Messages fail to be sent during data enqueuing.  Rectify the fault based on the error information in the ascend log."},
        {507052, "Memory copy fails during data enqueuing.  Rectify the fault based on the error information in the ascend log."},
        {507899, "An internal error occurs in the Driver module.  Rectify the fault based on the error information in the ascend log."},
        {507900, "An internal error occurs on the AI CPU module.  Rectify the fault based on the error information in the ascend log."},
        {507901, "The internal host device communication (HDC) session is disconnected.  Rectify the fault based on the error information in the ascend log."},
    }; /* aclError code */
};
}  // namespace c10_npu::acl
