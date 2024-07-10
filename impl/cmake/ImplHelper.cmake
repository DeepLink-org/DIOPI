macro(diopi_use_adapter cmd_extra_config)
    # dependency
    file(GLOB ADAPTOR_TEMPLATE_CODE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${ADAPTOR_DIR}/codegen/*.py)
    add_custom_target(adaptor_gen_dependency DEPENDS ${ADAPTOR_TEMPLATE_CODE})
    set(ADAPTOR_CSRC_PATH "${ADAPTOR_DIR}/csrc")

    separate_arguments(GenArgs UNIX_COMMAND "--diopi_dir=${DIOPI_IMPL_DIR}/../ --output_dir=${ADAPTOR_CSRC_PATH} ${cmd_extra_config}")
    message(STATUS "diopi_use_adapter GenArgs is:" "${GenArgs}")
    set(ADAPTER_GEN_FILES ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp ${ADAPTOR_CSRC_PATH}/impl_functions.hpp)
    add_custom_target(adaptor_code_gen
        COMMAND python3 ${ADAPTOR_DIR}/codegen/gen.py ${GenArgs}
        BYPRODUCTS ${ADAPTER_GEN_FILES}
        DEPENDS adaptor_gen_dependency
        VERBATIM
        )
    list(APPEND REAL_IMPL_SRC ${ADAPTOR_CSRC_PATH}/convert.cpp ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp ${ADAPTOR_CSRC_PATH}/composite_ops.cpp)
endmacro()


macro(prep_dyn_load if_dynload)
  if (${if_dynload})
    set(DYN_GEN_FILE  ${CMAKE_BINARY_DIR}/src/impl/wrap_function.cpp)
    set(DYN_HELP_DIR  ${DIOPI_IMPL_DIR}/scripts/dyn_load_helper)
    file(GLOB DYN_GEN_DEPS ${DYN_HELP_DIR}/dyn_wrap_gen.py)

    add_custom_target(dyn_wrap_gen ALL
             COMMAND python ${DYN_HELP_DIR}/dyn_wrap_gen.py -o ${DYN_GEN_FILE}
             DEPENDS ${DYN_GEN_DEPS}
             BYPRODUCTS ${DYN_GEN_FILE}
             WORKING_DIRECTORY ${DYN_HELP_DIR})
    set(DYN_SRC ${DYN_GEN_FILE} ${DYN_HELP_DIR}/dyn_helper.cpp)

    set(REAL_IMPL diopi_real_impl)
    add_library(${DEVICEIMPL} SHARED ${DYN_SRC})
    target_link_libraries(${DEVICEIMPL} -ldl)
    target_include_directories(${DEVICEIMPL} PRIVATE ${DYN_HELP_DIR})
    add_dependencies(${DEVICEIMPL} dyn_wrap_gen) 
  else()
    set(REAL_IMPL ${DEVICEIMPL})
  endif()
endmacro()

macro(handle_dyn_torch if_dynload torch_dir)
  if (${if_dynload})
    add_custom_target(dyn_torch
      COMMAND ${DIOPI_IMPL_DIR}/scripts/dyn_load_helper/dyn_torch_handler.sh patch_diopi
              ${LIBRARY_OUTPUT_PATH} ${torch_dir}/lib
      DEPENDS ${REAL_IMPL}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}

    )
    message(STATUS "handle_dyn_torch with torch: ${torch_dir}")
    add_dependencies(${DEVICEIMPL} dyn_torch)
  endif()
endmacro()

