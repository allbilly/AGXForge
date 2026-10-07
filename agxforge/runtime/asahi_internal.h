/* SPDX-License-Identifier: MIT */
#ifndef AGXFORGE_ASAHI_INTERNAL_H
#define AGXFORGE_ASAHI_INTERNAL_H
#include "asahi.h"
#include <drm/asahi_drm.h>
#include <stdbool.h>
#include <stddef.h>
#define AF_PAGE 16384ull
#define AF_ALIGN(v, a) (((v) + (a) - 1) & ~((a) - 1))
struct af_bo {
    struct af_device *dev;
    struct af_bo *next;
    uint64_t size, address;
    void *cpu;
    uint32_t handle, access;
    bool bound;
};
struct af_command {
    struct drm_asahi_cmd_header header;
    struct drm_asahi_cmd_compute compute;
};
struct af_device {
    int fd;
    char node[256], error[256];
    struct drm_asahi_params_global params;
    uint32_t major, minor, patch, vm_id, queue_id, sync_id;
    uint64_t usc_base, kernel_start, code_next, data_next;
    bool vm_live, queue_live, sync_live, prepared, poisoned;
    struct af_bo *bos, *usc, *cdm;
    uint32_t usc_size, cdm_size;
    struct af_command command;
};
int af_fail(struct af_device *d, const char *fmt, ...);
int af_ioctl(struct af_device *d, unsigned long request, void *arg, const char *name);
#endif
