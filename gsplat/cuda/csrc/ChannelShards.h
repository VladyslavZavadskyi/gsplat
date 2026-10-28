/*
 * SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstdint>
#include <type_traits>

#include "Config.h"
#include "Dispatch.h"

// =============================================================================
// Channel shards: compiling the per-channel kernel instantiations in parallel
//
// The rasterization kernels are instantiated once per channel count in
// GSPLAT_NUM_CHANNELS (times tile sizes and render options), so each of the
// RasterizeToPixels*.cu files used to be one long, single-threaded nvcc job.
//
// Each of those files is therefore compiled GSPLAT_NUM_CHANNEL_SHARDS times:
// X.cu itself is shard 0, and shard k is csrc/generated/X.shard<k>.cu, which
// defines GSPLAT_CHANNEL_SHARD and includes X.cu. build.py generates those for
// every .cu file that uses GSPLAT_CHANNEL_SHARD_NAME, so a file opts in just by
// following the pattern below. Shard k only instantiates the channel counts at
// positions p of GSPLAT_NUM_CHANNELS with p % GSPLAT_NUM_CHANNEL_SHARDS == k, so
// the shards compile in parallel while the set of kernels stays the same.
//
// A launcher `void launch_foo(...)` becomes one function per shard:
//   - X.cu defines it as `void GSPLAT_CHANNEL_SHARD_NAME(launch_foo)(...)`,
//     i.e. launch_foo_shard<k> in shard k, and dispatches over
//     channel_shards::Channels<GSPLAT_CHANNEL_SHARD> instead of all channels.
//   - The header declares `void launch_foo_shard0(...)` plus
//     `GSPLAT_DECLARE_CHANNEL_SHARDS(launch_foo);` for the other shards.
//   - Callers pick the shard for a runtime channel count:
//       launch_foo_shards[channel_shards::of(colors.size(-1))](...);
// Non-template functions with external linkage in X.cu must be compiled in
// shard 0 only (`#if GSPLAT_CHANNEL_SHARD == 0`), otherwise they're defined
// once per shard.
// =============================================================================

// The number of shards per sharded .cu file (X.cu + the generated X.shard<k>.cu);
// build.py reads it from this line. GSPLAT_DECLARE_CHANNEL_SHARDS below is
// written out for this value.
#define GSPLAT_NUM_CHANNEL_SHARDS 4

#ifndef GSPLAT_CHANNEL_SHARD
#    define GSPLAT_CHANNEL_SHARD 0
#endif

#define GSPLAT_CHANNEL_SHARD_CAT_(a, b) a##b
#define GSPLAT_CHANNEL_SHARD_CAT(a, b)  GSPLAT_CHANNEL_SHARD_CAT_(a, b)

// Name of this translation unit's shard of a launcher, e.g. launch_foo_shard2.
#define GSPLAT_CHANNEL_SHARD_NAME(fn) GSPLAT_CHANNEL_SHARD_CAT(fn##_shard, GSPLAT_CHANNEL_SHARD)

// Declares the shards 1..N-1 of a launcher (shard 0 must be declared before,
// with the full signature) and the table `fn_shards` indexed by shard.
#define GSPLAT_DECLARE_CHANNEL_SHARDS(fn)                                          \
    decltype(fn##_shard0) fn##_shard1, fn##_shard2, fn##_shard3;                   \
    inline constexpr decltype(&fn##_shard0) fn##_shards[GSPLAT_NUM_CHANNEL_SHARDS] \
        = {&fn##_shard0, &fn##_shard1, &fn##_shard2, &fn##_shard3}

static_assert(GSPLAT_NUM_CHANNEL_SHARDS == 4, "update GSPLAT_DECLARE_CHANNEL_SHARDS for the new shard count");
static_assert(
    GSPLAT_CHANNEL_SHARD >= 0 && GSPLAT_CHANNEL_SHARD < GSPLAT_NUM_CHANNEL_SHARDS, "GSPLAT_CHANNEL_SHARD out of range"
);

namespace gsplat::channel_shards
{
inline constexpr int kNumShards   = GSPLAT_NUM_CHANNEL_SHARDS;
inline constexpr int kChannels[]  = {GSPLAT_NUM_CHANNELS};
inline constexpr int kNumChannels = sizeof(kChannels) / sizeof(kChannels[0]);

// Shard holding the instantiations for a runtime channel count. Unsupported
// counts map to shard 0, whose dispatch then reports them as unsupported.
constexpr int of(int64_t channels)
{
    for(int i = 0; i < kNumChannels; ++i)
    {
        if(kChannels[i] == channels)
        {
            return i % kNumShards;
        }
    }
    return 0;
}

namespace detail
{
    // Collects the channel counts at positions Pos, Pos+1, ... (values C, Rest...)
    // that belong to Shard into the IntParam Acc.
    template<int Shard, typename Acc, int Pos, int... Values>
    struct Filter;

    template<int Shard, int... Acc, int Pos>
    struct Filter<Shard, dispatch::IntParam<Acc...>, Pos>
    {
        using type = dispatch::IntParam<Acc...>;
    };

    template<int Shard, int... Acc, int Pos, int C, int... Rest>
    struct Filter<Shard, dispatch::IntParam<Acc...>, Pos, C, Rest...>
    {
        using Next
            = std::conditional_t<Pos % kNumShards == Shard, dispatch::IntParam<Acc..., C>, dispatch::IntParam<Acc...>>;
        using type = typename Filter<Shard, Next, Pos + 1, Rest...>::type;
    };
} // namespace detail

// The channel counts instantiated by shard Shard, as a dispatch::IntParam.
// May be empty when GSPLAT_NUM_CHANNELS has fewer entries than shards.
template<int Shard>
using Channels = typename detail::Filter<Shard, dispatch::IntParam<>, 0, GSPLAT_NUM_CHANNELS>::type;
} // namespace gsplat::channel_shards
