/* SPDX-License-Identifier: MIT */
#ifndef AGXFORGE_ASAHI_H
#define AGXFORGE_ASAHI_H
#include <stdint.h>

/* Opaque ownership. No kernel structures cross this boundary. */
struct af_device;
struct af_bo;
struct af_device *af_open(const char *node, uint32_t va_slot);
int af_close(struct af_device *dev);
const char *af_error(struct af_device *dev);
const char *af_node(struct af_device *dev);
uint64_t af_param(struct af_device *dev, uint32_t key);
struct af_bo *af_alloc(struct af_device *dev, uint64_t bytes, uint32_t access);
int af_free(struct af_bo *bo);
void *af_map(struct af_bo *bo);
uint64_t af_address(struct af_bo *bo);
uint64_t af_size(struct af_bo *bo);
uint32_t af_handle(struct af_bo *bo);

/* access: bit 0 = GPU read, bit 1 = GPU write, bit 2 = USC address window. */
enum { AF_READ = 1, AF_WRITE = 2, AF_EXEC = 4 };
enum {
    AF_FEATURES, AF_GENERATION, AF_VARIANT, AF_REVISION, AF_CHIP,
    AF_VM_START, AF_VM_END, AF_KERNEL_MIN, AF_MAX_COMMANDS,
    AF_MAX_ATTACHMENTS, AF_TIMESTAMP_HZ, AF_DIES, AF_CLUSTERS,
    AF_CORES, AF_MAX_KHZ, AF_DRM_MAJOR, AF_DRM_MINOR, AF_DRM_PATCH,
    AF_USC_BASE, AF_KERNEL_START, AF_VM_ID, AF_QUEUE_ID,
    AF_PARAM_CORE_MASK = 64
};

/* Prepare separately so evidence can be saved BEFORE submission. */
int af_prepare(struct af_device *dev, struct af_bo *code, uint32_t entry,
               struct af_bo *uniforms, uint32_t uniform_offset, uint32_t uniform_halfs,
               uint32_t register_halfs, uint32_t global_x, uint32_t local_x);
int af_submit(struct af_device *dev, uint32_t timeout_ms);
/* Last launch state: 0 = USC, 1 = CDM, 2 = DRM command. */
const void *af_state(struct af_device *dev, uint32_t which);
uint32_t af_state_size(struct af_device *dev, uint32_t which);
uint64_t af_state_address(struct af_device *dev, uint32_t which);
#endif
