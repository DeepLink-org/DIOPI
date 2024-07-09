macro(diopi_use_adapter cmd_extra_config)
    # dependency
    file(GLOB ADAPTOR_TEMPLATE_CODE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${ADAPTOR_DIR}/codegen/*.py)
    add_custom_target(adaptor_gen_dependency DEPENDS ${ADAPTOR_TEMPLATE_CODE})
    set(ADAPTOR_CSRC_PATH "${ADAPTOR_DIR}/csrc")

    separate_arguments(GenArgs UNIX_COMMAND "--diopi_dir=${DIOPI_IMPL_DIR}/../ --output_dir=${ADAPTOR_CSRC_PATH} ${cmd_extra_config}")
    message(STATUS "diopi_use_adapter GenArgs is:" "${GenArgs}")
    set(GEN_FILES ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp ${ADAPTOR_CSRC_PATH}/impl_functions.hpp)
    add_custom_target(adaptor_code_gen
        COMMAND python3 ${ADAPTOR_DIR}/codegen/gen.py ${GenArgs}
        BYPRODUCTS ${GEN_FILES}
        DEPENDS adaptor_gen_dependency
        VERBATIM
        )
    list(APPEND REAL_IMPL_SRC ${ADAPTOR_CSRC_PATH}/convert.cpp ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp ${ADAPTOR_CSRC_PATH}/composite_ops.cpp)
endmacro()


macro(prep_dyn_load if_dynlod)
  if (${if_dynlod})
    # creat an empty file to pass wrap_func.cpp's existence check
    # one can change code_gen.py or wrap_func.cpp to recompile once wrap_func.cpp built
    execute_process(COMMAND touch ${CMAKE_CURRENT_SOURCE_DIR}/wrap_func.cpp)
    add_custom_target(dyn_wrap_gen COMMAND python ${DIOPI_IMPL_DIR}/scripts/dyn_load_helper/dyn_wrap_gen.py
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set(WRAPPER_SRC wrap_func.cpp)

    set(REAL_IMPL diopi_real_impl)
    add_library(${DEVICEIMPL} SHARED ${WRAPPER_SRC})
    target_link_libraries(${DEVICEIMPL} -ldl)
    add_dependencies(${DEVICEIMPL} dyn_wrap_gen) 
  else()
    set(REAL_IMPL ${DEVICEIMPL})
  endif()
endmacro()

# macro(prep_dync_load_torch)
  
# endmacro

