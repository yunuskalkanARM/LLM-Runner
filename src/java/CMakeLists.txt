#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# Declare project
project(arm-llm-java-prj
        VERSION     0.0.1
        DESCRIPTION "An LLM Java interface"
        LANGUAGES   C CXX ASM)

# Add the Java LLM API interface library. This is never going to be built
# but we can use its property to get the interface source used in other
# targets.
add_library(arm-llm-java INTERFACE)

if (${LLM_DEP_NAME} STREQUAL "llama.cpp")
    target_sources(arm-llm-java INTERFACE com/arm/llm/Llama.java com/arm/llm/LlamaConfig.java)
    add_dependencies(arm-llm-java arm-llm-jni)
#elseif(${LLM_DEP_NAME} STREQUAL "executorch")
#    # TODO
#elseif(${LLM_DEP_NAME} STREQUAL "mediapipe")
#    # TODO
else()
    message(FATAL_ERROR "${LLM_DEP_NAME} is currently not supported :(")
endif()
