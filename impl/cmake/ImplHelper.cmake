macro(diopi_use_adapter cmd_config)
    # dependency
    file(GLOB ADAPTOR_TEMPLATE_CODE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${ADAPTOR_DIR}/codegen/*.py)
    add_custom_target(adaptor_gen_dependency DEPENDS ${ADAPTOR_TEMPLATE_CODE})

    set(ADAPTOR_CSRC_PATH "${ADAPTOR_DIR}/csrc")
    set(GEN_FILES ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp ${ADAPTOR_CSRC_PATH}/impl_functions.hpp)
    add_custom_target(adaptor_code_gen
        COMMAND python3 ${ADAPTOR_DIR}/codegen/gen.py --diopi_dir=${DIOPI_IMPL_DIR}/../ --output_dir=${ADAPTOR_CSRC_PATH} ${cmd_config}
        BYPRODUCTS ${GEN_FILES}
        DEPENDS adaptor_gen_dependency)
    list(APPEND REAL_IMPL_SRC ${ADAPTOR_CSRC_PATH}/convert.cpp ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp ${ADAPTOR_CSRC_PATH}/composite_ops.cpp)
endmacro()

macro(prep_dyn_load if_dynlod)
  file(GLOB WRAPPER_SRC wrap_func.cpp)
  if (${if_dynlod})
    set(REAL_IMPL diopi_real_impl)
    add_library(${DEVICEIMPL} SHARED ${WRAPPER_SRC})
    target_link_libraries(${DEVICEIMPL} -ldl)
  else()
    set(REAL_IMPL ${DEVICEIMPL})
  endif()
endmacro()

# macro(prep_dync_load_torch)
  
# endmacro

