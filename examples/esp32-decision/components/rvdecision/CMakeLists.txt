idf_component_register(SRCS "rvdecision.c" INCLUDE_DIRS "include")
target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -fno-fast-math -Wall -Wextra)
# Halve wide accumulator additions on the 32-bit MCU targets. Native host
# builds keep the simple loop, which GCC can vectorize more effectively.
target_compile_definitions(${COMPONENT_LIB} PRIVATE RD_PAIR_DOT)
if(IDF_TARGET STREQUAL "esp32s3")
    # S3 instruction-count measurements favor newlib rounding. C6 and the
    # host use exact integer rounding to avoid software float conversions.
    target_compile_definitions(${COMPONENT_LIB} PRIVATE RD_LIBM_ROUND)
endif()
