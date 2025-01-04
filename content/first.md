+++
title = "How to use distributed shared memory in CUDA for inter-thread-block communication"
date = 2025-04-01
+++

# Intro

I noticed there is very little information or easy examples for using the sm-to-sm communication network on the Hopper Architecture (Compute Capability 9.0/`sm_90`),
so i thought i would try to quickly give a super simple example. Ive also been working on a more extended writeup on using the distributed shared memory features for a
NBODY-simulation kernel, but that will probably have to wait till im done with my exams. Also this is my first attempt at a blog, so im super grateful for any feedback.

## What is distributed shared memory ?

So what is distributed shared memory (DSMEM) and why do we want it ? Most "interesting" GPU-Problems end up being very sensitiv to memory characterstics (latency/bandwidth),
so much so that in my GPU architecture course we ended almost exclusively talking about the memory-model of CUDA and hardware implementation of it.
A real quick breakdown of it goes like this:

1. You have global memory which is usually what the number is you see for GPU-memory on the label. This ends up being DRAM since we want loads of it to store our results
   (just look at the size's of those LLamas3.1 Models), but that also means we hit the 'memory wall' and end up being bottlenecked fairly early by memory-speeds.
1. There's also constant/texture memory which i won't go much into in here, but basiaclly its just global memory with its own cache (also DRAM).
1. To make my point im also skipping all the caches, and registers.
1. Shared memory, which is essentially a programmer-controlled cache that resides on the silicon itself and consists of SRAM which ends up being a whole bunch faster (roughly 10-20x better latency and 2-3x bandwith).
   To achieve the advertised FLOPS modern GPU's can put out, your code will certainly have to make use of this to recycle data-accesses. One big caveat with this is though, that (sofar) shared-memory
   is only accessable by threads in the same thread-block. So while you can have up-to 1024 threads work on the same chunk of shared-memory, if you want to access it from outside that thread-block,
   so far you would need to copy the content's back into global memory. One can easily see how this might really hurt us on performance, since the whole reason we are using shared-memory is because we want to
   avoid those time/energy cost's of going to device/global-memory.

So what the fine people at NVIDIA did with their Hopper Architecture is add a new address-space to the memory-model, which let's us access a different's thread-block memory without going to main memory.
This is implemented via some inter-streaming-multiprocessor communication-net the details of which seem to still be obscured, but that also doesn't matter too much for us. It is to note though that with the current
architecture CUDA only supports this for thread-blocks that are in the same thread-block-cluster (clusters are limited to a total size of 16 blocks).

## How do we actually use the new feature ?

So first of, as mentioned you won't be able to use this feature if you're not running on an Compute Capability 9.0 (or higher if youre reading post blackwell release) CUDA-Card (i.e H100/H200),
this also means that we need to compile our CUDA-kernel to be specific to the architecture (we can do this by passing `-arch=sm_90`) since otherwise nvcc will compile our kernel for a bunch of
different target-architectures (which is called a fat-binary internaly).

Then our next condition is to launch our kernel via the extended kernel-launch interface `cudaLaunchKernelEx(...)` (see [here](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g98d60efe48c3400a1c17a1edb698e530)),
not the standard tripple angle-brackets. This is nothing complicated, just way more verbose. Crucially we have to specify the thread-block-cluster size to make sure we are running our kernel in a thread-block-cluster mode.

```cpp
// our config object
cudaLaunchConfig_t config = {0};
// other arguments

// new attribute 
cudaLaunchAttribute attribute[1]; // only one attribute in this case
attribute[0].id = cudaLaunchAttributeClusterDimension; // specify attribute type
attribute[0].val.clusterDim.x = cluster_size; // our cluster size (up to 16)

// add our attribute to the config
config.numAttrs = 1;
config.attrs = attribute;

cudaLaunchKernelEx(&config, kernel); // launch our kernel with the config
```

Now finally we can get to the feature, accessing another thread-blocks local/shared memory.
For this we need a handle for the cluster we (the current thread/warp/thread-block) is inside which we get by calling `cooperative_groups::this_cluster()` (dont forget to `#include <cooperative_groups.h>`).
Then using this `cluster.map_shared_rank()` ( [see here for documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cluster-group-cg)) we can get a pointer to a shared-memory variable that we want from the remote thread-block:

```cpp
// some variable in shared-memory
extern __shared__ int A;

auto cluster = cooperative_groups::this_cluster();

// get address of A variable of second thread-block in the cluster
int* remote_A = cluster.map_shared_rank(&A, 1);

*remote_A += 0xBEEF; // modify remote memory
```

This ends up doing exactly what we want, but lets convince ourselfs it actually does what we want with ...

## A 'hello world' level DSMEM program

So to demonstrate this actually works, i tried to come up with a super simple example kernel, which is purely contrived for seeing that the feature works:

```cpp
__global__ void sm2smTest() {
  // handle object to the current thread-block-cluster
  cooperative_groups::cluster_group cluster =
      cooperative_groups::this_cluster();

  // declare and initalize our shared memory
  extern __shared__ int smem[];
  smem[threadIdx.x] = blockIdx.x * threadIdx.x;

  // cluster-wide barrier to ensure all the shared-memory is initalized
  cluster.sync();

  // aquire address of the 'smem' variable of the next thread-block (wrap in )
  int *dst_smem =
      cluster.map_shared_rank(smem, (blockIdx.x + 1) % cluster.dim_blocks().x);

  // write from our local thread-block to the remote block memory
  dst_smem[threadIdx.x] += 1;

  // another barrier to ensure all remote thread-blocks are done writing to our
  // smem
  cluster.sync();

  // inital value in smem should've been incremented by neighbour thread-block
  // so smem[threadIdx.x] = (threadIdx.x * blockIdx.x) + 1
  printf("thread-idx: %d\tblock-idx: %d\tsmem: %d\n", threadIdx.x, blockIdx.x,
         smem[threadIdx.x]);
}
```

(_Note:_ Dont forget to call `cluster.sync()` at the appropriate spots to ensure our operations are all synchronized correctly).

When launched via the `cudaLaunchKernelEx` with 4 blocks and one single cluster we get something like:

```bash
thread-idx: 0	block-idx: 0	smem: 1
thread-idx: 1	block-idx: 0	smem: 1
thread-idx: 2	block-idx: 0	smem: 1
...
thread-idx: 0	block-idx: 1	smem: 1
thread-idx: 1	block-idx: 1	smem: 2
thread-idx: 2	block-idx: 1	smem: 3
... 
thread-idx: 0	block-idx: 2	smem: 1
thread-idx: 1	block-idx: 2	smem: 3
thread-idx: 2	block-idx: 2	smem: 5
...

thread-idx: 0	block-idx: 3	smem: 1
thread-idx: 1	block-idx: 3	smem: 4
thread-idx: 2	block-idx: 3	smem: 7
thread-idx: 3	block-idx: 3	smem: 10
...
```

Yes this a trivial example, but we can see that the `dst_smem[threadIdx.x] += 1` is actually happening and being executed, success 🎉.
The next step is comming up with algorithms that can exploit this new feature properly, which isn't necessarily trivial. In my experience the extra synchronization that's needed often makes up any
performance gained from using DSMEM, so this feature is probably best used in-cases where synchronization had to happen anyways.

## Conclusion

This was ment to be only a super short and easy intro to using this new feature, for more information you can look at the documentation i linked above or read the more verbose [offical-guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#distributed-shared-memory).
If you have any questions feel free to reach out to me on [bluesky](https://bsky.app/profile/jakobs99.bsky.social), or leave a comment on the hackernews thread.
