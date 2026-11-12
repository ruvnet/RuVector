idf_component_register(SRCS "${RD_KERNEL_SOURCE}" INCLUDE_DIRS "include")
if(RD_MODEL_SIZED_WORKSPACE)
    include("${CMAKE_CURRENT_LIST_DIR}/model_capacity.cmake")
    rd_model_capacity("${RD_MODEL_DIR}/model.h" RD_CAP_DIMS RD_CAP_CLASSES)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${RD_MODEL_DIR}/model.h")
    # PUBLIC is required: rd_workspace, rd_result and rd_context share this ABI
    # with main/app/profile and any future component that requires rvdecision.
    target_compile_definitions(${COMPONENT_LIB} PUBLIC
        RD_MAX_DIMS=${RD_CAP_DIMS} RD_MAX_CLASSES=${RD_CAP_CLASSES})
endif()
file(SHA256 "${RD_KERNEL_SOURCE}" RD_KERNEL_HASH)
target_compile_definitions(${COMPONENT_LIB} PUBLIC RD_KERNEL_SHA256="${RD_KERNEL_HASH}")
target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -fno-fast-math -ffp-contract=off -Wall -Wextra)
# Halve wide accumulator additions on the 32-bit MCU targets. Native host
# builds keep the simple loop, which GCC can vectorize more effectively.
target_compile_definitions(${COMPONENT_LIB} PRIVATE RD_PAIR_DOT)
if(IDF_TARGET STREQUAL "esp32s3")
    # S3 instruction-count measurements favor newlib rounding. C6 and the
    # host use exact integer rounding to avoid software float conversions.
    target_compile_definitions(${COMPONENT_LIB} PRIVATE RD_LIBM_ROUND)
endif()
