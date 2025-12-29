#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited
#
# SPDX-License-Identifier: Apache-2.0
#
include_guard(GLOBAL)

function(arm_enable_streamline target_name)

    message(STATUS "STREAMLINE ENABLED for target: ${target_name}")
    include(FetchContent)

    # ---------------------------------------------------------------------------
    # Fetch Arm Gator sources for Streamline annotations
    # ---------------------------------------------------------------------------
    set(GATOR_SRC_DIR "${CMAKE_BINARY_DIR}/gator"
      CACHE PATH "Streamline annotate source dir")

    set(GATOR_GIT_URL "https://github.com/ARM-software/gator.git"
            CACHE STRING "Git URL for Gator repo")

    #Gator 9.7.2 compatible with Arm Performance Studio 2025.6
    set(GATOR_GIT_TAG "f0774012f36dbdb543e082d3e14ca9db20d0432d"
            CACHE STRING "Git tag / commit SHA for Gator repo")

    FetchContent_Declare(streamline_annotate_src
        GIT_REPOSITORY ${GATOR_GIT_URL}
        GIT_TAG        ${GATOR_GIT_TAG}
        GIT_SHALLOW    1
        SOURCE_DIR     ${GATOR_SRC_DIR}
        EXCLUDE_FROM_ALL
    )

    FetchContent_Populate(streamline_annotate_src)

    # ---------------------------------------------------------------------------
    # Implementation library (contains gator_annotate_* symbols)
    # ---------------------------------------------------------------------------
    if(NOT TARGET streamline_annotate)
        add_library(streamline_annotate STATIC
            ${GATOR_SRC_DIR}/annotate/streamline_annotate.c
        )

        target_include_directories(streamline_annotate PUBLIC
            ${GATOR_SRC_DIR}/annotate
        )

        set_property(TARGET streamline_annotate
            PROPERTY POSITION_INDEPENDENT_CODE ON)
    endif()

    # ---------------------------------------------------------------------------
    # Feature bundle: everything a target needs to *use* Streamline
    # ---------------------------------------------------------------------------
    if(NOT TARGET arm_streamline)
        add_library(arm_streamline INTERFACE)

        # Link implementation
        target_link_libraries(arm_streamline INTERFACE
            streamline_annotate
        )

        # Enable annotations in code
        target_compile_definitions(arm_streamline INTERFACE
            ENABLE_STREAMLINE
        )

        # Required for usable callstacks
        target_compile_options(arm_streamline INTERFACE
            -g
            -fno-omit-frame-pointer
            -fno-inline
        )

        # Make #include "profiling/StreamlineLlm.hpp" work
        target_include_directories(arm_streamline INTERFACE
            ${PROJECT_SOURCE_DIR}/src
        )
    endif()

    # ---------------------------------------------------------------------------
    # Apply Streamline to the requested target
    # ---------------------------------------------------------------------------
    target_link_libraries(${target_name} PUBLIC arm_streamline)

    set_property(TARGET ${target_name}
        PROPERTY POSITION_INDEPENDENT_CODE ON)

endfunction()
