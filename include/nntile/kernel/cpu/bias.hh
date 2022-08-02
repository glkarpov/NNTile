/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/cpu/bias.hh
 * Bias operation on a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-02
 * */

#include <nntile/base_types.hh>

namespace nntile
{

// Apply bias along middle axis
template<typename T>
void bias_kernel_cpu(Index m, Index n, Index k, const T *src, T *dst)
    noexcept;

// Apply bias along middle axis of StarPU buffer
template<typename T>
void bias_starpu_cpu(void *buffers[], void *cl_args)
    noexcept;

} // namespace nntile

