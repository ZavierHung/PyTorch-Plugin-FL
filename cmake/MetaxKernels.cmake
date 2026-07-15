# Copyright (c) 2026, BAAI. All rights reserved.
# Compile MetaX device kernels with mxcc/cucc (no CMake CUDA language).

function(flagos_add_metax_kernel_objects out_var backend_dir kernel_list)
  if(NOT METAX_PATH)
    set(_metax_path "$ENV{METAX_PATH}")
  endif()
  if(NOT _metax_path)
    if(DEFINED ENV{MACA_PATH})
      set(_metax_path "$ENV{MACA_PATH}")
    elseif(DEFINED ENV{MACA_HOME})
      set(_metax_path "$ENV{MACA_HOME}")
    else()
      set(_metax_path "/opt/maca")
    endif()
  endif()

  set(_cu_bridge "${_metax_path}/tools/cu-bridge")
  set(_metax_cc "${_cu_bridge}/bin/cucc")
  if(DEFINED ENV{METAX_MXCC})
    set(_metax_cc "$ENV{METAX_MXCC}")
  elseif(DEFINED ENV{METAX_CUCC})
    set(_metax_cc "$ENV{METAX_CUCC}")
  endif()

  if(NOT EXISTS "${_metax_cc}")
    message(FATAL_ERROR "MetaX compiler not found: ${_metax_cc}")
  endif()

  set(_arch "80")
  if(DEFINED ENV{METAX_ARCH})
    set(_arch "$ENV{METAX_ARCH}")
  elseif(DEFINED ENV{MACA_ARCH})
    set(_arch "$ENV{MACA_ARCH}")
  endif()

  set(_inc
    -I${CMAKE_SOURCE_DIR}
    -I${CMAKE_SOURCE_DIR}/csrc
    -I${CMAKE_SOURCE_DIR}/csrc/aten/backends/metax
    -I${CMAKE_SOURCE_DIR}/csrc/aten/backends/${backend_dir}
    -I${PYTORCH_INSTALL_DIR}/include
    -I${_cu_bridge}/include
    -I${_metax_path}/include
    -I${_metax_path}/include/mcr
  )

  set(_flags -std=c++17 -fPIC -O3 -DMETAX_ARCH=${_arch} -DUSE_MACA=1)

  set(_objs)
  set(_obj_dir "${CMAKE_CURRENT_BINARY_DIR}/metax_kernels_${backend_dir}")
  file(MAKE_DIRECTORY "${_obj_dir}")

  foreach(_k IN LISTS kernel_list)
    set(_cu "${CMAKE_SOURCE_DIR}/csrc/aten/backends/${backend_dir}/${_k}.cu")
    if(NOT EXISTS "${_cu}")
      message(FATAL_ERROR "MetaX kernel source not found: ${_cu}")
    endif()
    get_filename_component(_stem "${_k}" NAME_WE)
    set(_obj "${_obj_dir}/${_k}.o")
    set(_extra_deps
      "${CMAKE_SOURCE_DIR}/csrc/aten/backends/metax/metax_elementwise.cuh")
    if(backend_dir STREQUAL "metax")
      list(APPEND _extra_deps
        "${CMAKE_SOURCE_DIR}/csrc/aten/backends/metax/${_stem}_kernel.cuh"
        "${CMAKE_SOURCE_DIR}/csrc/aten/backends/metax/metax_elementwise.cuh")
    elseif(backend_dir STREQUAL "metax_opt")
      string(REGEX REPLACE "_opt$" "" _base "${_stem}")
      list(APPEND _extra_deps
        "${CMAKE_SOURCE_DIR}/csrc/aten/backends/metax_opt/metax_elementwise_opt.cuh")
      if(EXISTS "${CMAKE_SOURCE_DIR}/csrc/aten/backends/metax_opt/${_base}_kernel_opt.cuh")
        list(APPEND _extra_deps
          "${CMAKE_SOURCE_DIR}/csrc/aten/backends/metax_opt/${_base}_kernel_opt.cuh")
      endif()
    elseif(backend_dir STREQUAL "metax_v2")
      string(REGEX REPLACE "_v2$" "" _base "${_stem}")
      list(APPEND _extra_deps
        "${CMAKE_SOURCE_DIR}/csrc/aten/backends/metax_v2/metax_elementwise_v2.cuh")
      if(EXISTS "${CMAKE_SOURCE_DIR}/csrc/aten/backends/metax_v2/${_base}_kernel_v2.cuh")
        list(APPEND _extra_deps
          "${CMAKE_SOURCE_DIR}/csrc/aten/backends/metax_v2/${_base}_kernel_v2.cuh")
      endif()
    endif()
    add_custom_command(
      OUTPUT "${_obj}"
      COMMAND ${_metax_cc} -c ${_cu} -o ${_obj} ${_inc} ${_flags}
      DEPENDS "${_cu}" ${_extra_deps}
      COMMENT "MetaX mxcc [${backend_dir}]: ${_k}.cu"
      VERBATIM
    )
    list(APPEND _objs "${_obj}")
  endforeach()

  set(${out_var} ${_objs} PARENT_SCOPE)
endfunction()

function(flagos_add_all_metax_kernel_objects out_var)
  set(_baseline_kernels add mul le all neg embedding mean mm bmm cos sin rsqrt silu pow sum ones_like softmax where bitwise_and)
  set(_opt_kernels add_opt mul_opt mean_opt mm_opt bmm_opt pow_opt rsqrt_opt)
  set(_v2_kernels add_v2 mul_v2 mean_v2 mm_v2 bmm_v2 pow_v2 rsqrt_v2 rmsnorm_v2 rope_v2 softmax_v2 swiglu_v2 add_rms_norm_v2 copy_v2 cat_v2_fused elementwise_vec_v2 elementwise_offset_v2 elementwise_broadcast_v2)

  flagos_add_metax_kernel_objects(_baseline_objs metax "${_baseline_kernels}")
  flagos_add_metax_kernel_objects(_opt_objs metax_opt "${_opt_kernels}")
  flagos_add_metax_kernel_objects(_v2_objs metax_v2 "${_v2_kernels}")

  set(${out_var} ${_baseline_objs} ${_opt_objs} ${_v2_objs} PARENT_SCOPE)
endfunction()
