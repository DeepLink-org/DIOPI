function(diopi_use_adapter adaptor_dir diopi_impl_dir config_device base_device out_src_files)
    # NB: all augements passed by func parameters instead of global variables.
    file(GLOB ADAPTOR_TEMPLATE_CODE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${adaptor_dir}/codegen/*.py)
    add_custom_target(adaptor_gen_dependency DEPENDS ${ADAPTOR_TEMPLATE_CODE})
    set(ADAPTOR_CSRC_PATH "${ADAPTOR_DIR}/csrc")

    set(ADAPTER_GEN_FILES ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp ${ADAPTOR_CSRC_PATH}/impl_functions.hpp)
    add_custom_target(adaptor_code_gen
        COMMAND python3 ${ADAPTOR_DIR}/codegen/gen.py --diopi_dir=${diopi_impl_dir}/../ --output_dir=${ADAPTOR_CSRC_PATH} --config_device=${config_device} --base_device=${base_device}
        BYPRODUCTS ${ADAPTER_GEN_FILES}
        DEPENDS adaptor_gen_dependency
        VERBATIM
        )
    list(APPEND ${out_src_files} ${ADAPTOR_CSRC_PATH}/convert.cpp ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp ${ADAPTOR_CSRC_PATH}/composite_ops.cpp)
endfunction()


function(prep_dyn_load diopi_impl_dir device_impl)
    set(DYN_GEN_FILE  ${CMAKE_BINARY_DIR}/src/impl/wrap_function.cpp)
    set(DYN_HELP_DIR  ${diopi_impl_dir}/scripts/dyn_load_helper)
    file(GLOB DYN_GEN_DEPS ${DYN_HELP_DIR}/dyn_wrap_gen.py)

    add_custom_target(dyn_wrap_gen ALL
             COMMAND python ${DYN_HELP_DIR}/dyn_wrap_gen.py -o ${DYN_GEN_FILE}
             DEPENDS ${DYN_GEN_DEPS}
             BYPRODUCTS ${DYN_GEN_FILE}
             WORKING_DIRECTORY ${DYN_HELP_DIR})
    set(DYN_SRC ${DYN_GEN_FILE} ${DYN_HELP_DIR}/dyn_helper.cpp)

    add_library(${device_impl} SHARED ${DYN_SRC})
    target_link_libraries(${device_impl} -ldl)
    target_include_directories(${device_impl} PRIVATE ${DYN_HELP_DIR})
    add_dependencies(${device_impl} dyn_wrap_gen)
endfunction()

function(handle_dyn_torch diopi_impl_dir real_impl torch_dir device_impl)
    add_custom_target(dyn_torch
      COMMAND ${diopi_impl_dir}/scripts/dyn_load_helper/dyn_torch_handler.sh patch_diopi
            ${LIBRARY_OUTPUT_PATH} ${torch_dir}/lib
      DEPENDS ${real_impl}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    message(STATUS "handle_dyn_torch with torch: ${torch_dir}")
    add_dependencies(${device_impl} dyn_torch)
endfunction()

