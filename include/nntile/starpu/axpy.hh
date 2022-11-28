/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/axpy.hh
 * AXPY operation for StarPU buffers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-11-23
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/constants.hh>
// This also includes all definitions
#include <nntile/starpu/config.hh>

namespace nntile
{
namespace starpu
{
namespace axpy
{

//! Structure for arguments
template<typename T>
struct args2_t
{
    Index nelems;
    T alpha;
};

#ifdef NNTILE_USE_CBLAS
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept;

template<typename T>
void cpu2(void *buffers[], void *cl_args)
    noexcept;
#endif // NNTILE_USE_CBLAS

#ifdef NNTILE_USE_CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept;

template<typename T>
void cuda2(void *buffers[], void *cl_args)
    noexcept;
#endif // NNTILE_USE_CUDA

extern Codelet codelet_fp32, codelet_fp64, codelet2_fp32, codelet2_fp64;

template<typename T>
constexpr Codelet *codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr Codelet *codelet<fp32_t>()
{
    return &codelet_fp32;
}

template<>
constexpr Codelet *codelet<fp64_t>()
{
    return &codelet_fp64;
}

template<typename T>
constexpr Codelet *codelet2()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr Codelet *codelet2<fp32_t>()
{
    return &codelet2_fp32;
}

template<>
constexpr Codelet *codelet2<fp64_t>()
{
    return &codelet2_fp64;
}

void init();

void restrict_where(uint32_t where);

void restore_where();

template<typename T>
void submit(Handle alpha, Index nelems, Handle src, Handle dst);

template<typename T>
void submit2(T alpha, Index nelems, Handle src, Handle dst);

} // namespace axpy
} // namespace starpu
} // namespace nntile

