//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#ifndef LLM_STREAMLINE_HPP
#define LLM_STREAMLINE_HPP

#include "streamline_annotate.h"
#include <cstdint>

namespace sl {

/**
 * @brief Stable group/channel identifiers for LLM annotations.
 *
 * Keep these IDs stable to make captures comparable across runs.
 */
enum : uint32_t {
    GROUP_LLM      = 1,     ///< Streamline group for LLM annotations
    CH_INIT        = 10,    ///< Model/runtime initialization
    CH_ENCODE      = 11,    ///< Prompt processing / prefill
    CH_DECODE      = 12,    ///< Token generation loop
    CH_CONTROL     = 13,    ///< Cancellation / stop / teardown paths
};

/**
 * @brief Initialize Streamline annotation naming for the current thread.
 *
 * Streamline defines groups/channels per thread. This function calls
 * ANNOTATE_SETUP and names the LLM group and channels exactly once per thread.
 *
 * Safe to call repeatedly; subsequent calls on the same thread are no-ops.
 */
inline void InitThreadOnce()
{
    static thread_local bool inited = false;
    if (inited) return;

    ANNOTATE_SETUP;

    // Per-thread naming (required by Streamline)
    ANNOTATE_NAME_GROUP(GROUP_LLM, "LLM");
    ANNOTATE_NAME_CHANNEL(CH_INIT,        GROUP_LLM, "Init");
    ANNOTATE_NAME_CHANNEL(CH_ENCODE,      GROUP_LLM, "Encode");
    ANNOTATE_NAME_CHANNEL(CH_DECODE,      GROUP_LLM, "Decode");
    ANNOTATE_NAME_CHANNEL(CH_CONTROL,     GROUP_LLM, "Control");

    inited = true;
}

/**
 * @brief RAII helper for timed Streamline channel annotations.
 *
 * Creates a channel annotation on construction and ends it on destruction.
 * This guarantees correct begin/end pairing even with early returns.
 */
struct Scope {
    uint32_t ch;    ///< Channel used by this scope

    /**
     * @brief Begin a timed annotation on the given channel.
     * @param channel Streamline channel ID (e.g. CH_ENCODE).
     * @param color   Streamline color constant (e.g. ANNOTATE_GREEN).
     * @param name    Label shown in the Streamline GUI.
     */
    Scope(uint32_t channel, uint32_t color, const char* name) : ch(channel)
    {
        ANNOTATE_CHANNEL_COLOR(ch, color, name);
    }

     /**
     * @brief End the timed annotation for this channel.
     */
    ~Scope()
    {
        ANNOTATE_CHANNEL_END(ch);
    }
};

/**
 * @brief Emit a point-in-time marker event.
 *
 * Markers are intended for rare, high-signal events (e.g. cancel/stop requests),
 * not for hot paths (e.g. per-token loops).
 *
 * @param color Streamline color constant.
 * @param text  Marker label shown in the Streamline GUI.
 */
inline void marker(uint32_t color, const char* text)
{
    ANNOTATE_MARKER_COLOR_STR(color, text);
}

} // namespace sl
#endif /* LLM_STREAMLINE_HPP */
