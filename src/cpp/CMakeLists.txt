#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# Declare project
project(arm-llm-cpp-prj
        VERSION     0.0.1
        DESCRIPTION "An LLM CPP interface"
        LANGUAGES   C CXX ASM)

# Add the LLM API interface library
# NOTE: This is an interface library right now because our wrapper is
#       only a header file. If we need to add source files later, this
#       would change to a SHARED or STATIC library.
add_library(arm-llm-cpp INTERFACE)
target_include_directories(arm-llm-cpp INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (${LLM_DEP_NAME} STREQUAL "llama.cpp")
    add_subdirectory(llama_cpp)
#elseif(${LLM_DEP_NAME} STREQUAL "executorch")
#    # TODO
#elseif(${LLM_DEP_NAME} STREQUAL "mediapipe")
#    # TODO
else()
    message(FATAL_ERROR "${LLM_DEP_NAME} is currently not supported :(")
endif()
