idf_component_register(SRCS "rvdecision.c" INCLUDE_DIRS "include")
target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -fno-fast-math -Wall -Wextra)
