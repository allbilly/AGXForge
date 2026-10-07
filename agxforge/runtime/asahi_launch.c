/* SPDX-License-Identifier: MIT */
/* Packet definitions: Copyright 2021-2025 Alyssa Rosenzweig;
 * Copyright 2023-2025 Valve Corporation (Mesa cmdbuf.xml). */
/* Restricted G13G USC/CDM packets; fields from pinned Mesa cmdbuf.xml.
 * See experimental/asahi-launch/sources.json for provenance. */
#define _POSIX_C_SOURCE 200809L
#include "asahi_internal.h"
#include <drm/drm.h>
#include <limits.h>
#include <stdatomic.h>
#include <string.h>
#include <time.h>

static void put16(uint8_t **p, uint16_t value) { memcpy(*p, &value, 2); *p += 2; }
static void put32(uint8_t **p, uint32_t value) { memcpy(*p, &value, 4); *p += 4; }
static void put64(uint8_t **p, uint64_t value) { memcpy(*p, &value, 8); *p += 8; }

int af_prepare(struct af_device *d, struct af_bo *code, uint32_t entry,
               struct af_bo *uniforms, uint32_t uniform_offset, uint32_t uniform_halfs,
               uint32_t register_halfs, uint32_t global_x, uint32_t local_x)
{
    d->prepared = false;
    if (d->poisoned) return af_fail(d, "device stopped after failed completion");
    if (!code || !uniforms || code->dev != d || uniforms->dev != d ||
        !(code->access & AF_EXEC) || !(code->access & AF_READ) ||
        !(uniforms->access & AF_READ) || entry % 2 || entry >= code->size ||
        !uniform_halfs || uniform_halfs > 256 || uniform_offset % 4 ||
        uniform_offset >= uniforms->size || uniform_halfs * 2 > uniforms->size - uniform_offset ||
        !register_halfs || register_halfs > 248 || !global_x || global_x > INT_MAX ||
        !local_x || local_x > 1024 || local_x % 32 || global_x % local_x)
        return af_fail(d, "launch violates G13G scalar contract");
    if (!d->usc) d->usc = af_alloc(d, AF_PAGE, AF_READ | AF_EXEC);
    if (!d->usc) return -1;
    if (!d->cdm) d->cdm = af_alloc(d, AF_PAGE, AF_READ);
    if (!d->cdm) return -1;
    uint8_t *p = d->usc->cpu;
    memset(p, 0, d->usc->size);
    for (uint32_t start = 0; start < uniform_halfs; start += 64) {
        uint32_t count = uniform_halfs - start;
        if (count > 64) count = 64;
        uint64_t addr = uniforms->address + uniform_offset + start * 2;
        if (addr >= (1ull << 40) || addr % 4) return af_fail(d, "uniform pointer out of USC range");
        put64(&p, 0x1dull | ((uint64_t)start << 8) | ((uint64_t)(count & 63) << 20) | ((addr >> 2) << 26));
    }
    put32(&p, 0x904d); /* SHARED: vertex/compute layout, uses_shared_memory=false */
    put16(&p, 0x0c0d); /* SHADER: unk_2 = 3, loads_varyings=false */
    put32(&p, (uint32_t)(code->address + entry - d->usc_base));
    put32(&p, 0x8d | (((register_halfs + 7) / 8) << 8));
    put16(&p, 0x88); /* NO_PRESHADER */
    d->usc_size = p - (uint8_t *)d->usc->cpu;
    p = d->cdm->cpu;
    memset(p, 0, d->cdm->size);
    put32(&p, ((uniform_halfs + 63) / 64) << 1);
    put32(&p, (uint32_t)(d->usc->address - d->usc_base));
    put32(&p, global_x); put32(&p, 1); put32(&p, 1);
    put32(&p, local_x); put32(&p, 1); put32(&p, 1);
    put32(&p, 0x600fffff); /* G13G cache/visibility barrier from Mesa */
    put32(&p, 0x40000000); put32(&p, 0); /* padded STREAM_TERMINATE */
    d->cdm_size = p - (uint8_t *)d->cdm->cpu;
    memset(&d->command, 0, sizeof(d->command));
    d->command.header.cmd_type = DRM_ASAHI_CMD_COMPUTE;
    d->command.header.size = sizeof(d->command.compute);
    /* Both zero barriers wait for previous work on this queue. */
    d->command.compute.cdm_ctrl_stream_base = d->cdm->address;
    d->command.compute.cdm_ctrl_stream_end = d->cdm->address + d->cdm_size - 8;
    d->prepared = true;
    return 0;
}

int af_submit(struct af_device *d, uint32_t timeout_ms)
{
    if (!d->prepared || d->poisoned || !timeout_ms || timeout_ms > 60000)
        return af_fail(d, "no prepared launch or invalid completion timeout");
    d->prepared = false;
    if (d->sync_live) {
        struct drm_syncobj_destroy old = {.handle = d->sync_id};
        if (af_ioctl(d, DRM_IOCTL_SYNCOBJ_DESTROY, &old, "SYNCOBJ_DESTROY")) return -1;
        d->sync_live = false;
    }
    struct drm_syncobj_create create = {0};
    if (af_ioctl(d, DRM_IOCTL_SYNCOBJ_CREATE, &create, "SYNCOBJ_CREATE")) return -1;
    d->sync_id = create.handle; d->sync_live = true;
    struct drm_asahi_sync sync = {.sync_type = DRM_ASAHI_SYNC_SYNCOBJ, .handle = d->sync_id};
    struct drm_asahi_submit submit = {
        .syncs = (uintptr_t)&sync, .cmdbuf = (uintptr_t)&d->command,
        .queue_id = d->queue_id, .out_sync_count = 1, .cmdbuf_size = sizeof(d->command)
    };
    atomic_thread_fence(memory_order_seq_cst);
    if (af_ioctl(d, DRM_IOCTL_ASAHI_SUBMIT, &submit, "SUBMIT")) return -1;
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now)) { d->poisoned = true; return af_fail(d, "clock_gettime failed"); }
    struct drm_syncobj_wait wait = {
        .handles = (uintptr_t)&d->sync_id, .count_handles = 1,
        .timeout_nsec = (int64_t)now.tv_sec * 1000000000 + now.tv_nsec + (int64_t)timeout_ms * 1000000,
        .flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL
    };
    if (af_ioctl(d, DRM_IOCTL_SYNCOBJ_WAIT, &wait, "SYNCOBJ_WAIT")) { d->poisoned = true; return -1; }
    atomic_thread_fence(memory_order_seq_cst);
    return 0; /* Completion alone is not an output correctness result. */
}

const void *af_state(struct af_device *d, uint32_t which)
{
    if (which == 0) return d->usc ? d->usc->cpu : NULL;
    if (which == 1) return d->cdm ? d->cdm->cpu : NULL;
    return which == 2 ? &d->command : NULL;
}
uint32_t af_state_size(struct af_device *d, uint32_t which)
{
    return which == 0 ? d->usc_size : which == 1 ? d->cdm_size : which == 2 ? sizeof(d->command) : 0;
}
uint64_t af_state_address(struct af_device *d, uint32_t which)
{
    return which == 0 && d->usc ? d->usc->address : which == 1 && d->cdm ? d->cdm->address : 0;
}
