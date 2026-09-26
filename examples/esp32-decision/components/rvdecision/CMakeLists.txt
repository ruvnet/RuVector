idf_component_register(SRCS "${RD_KERNEL_SOURCE}" INCLUDE_DIRS "include")
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
