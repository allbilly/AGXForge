# macOS G13 source references

The assembler and disassembler use the existing applegpu tables and
[BSD 3-Clause attribution](../asahi-launch/THIRD-PARTY-NOTICES.md).
The Metal binary-archive replacement workflow was studied in Dougall Johnson's
[applegpu hardware harness](https://github.com/dougallj/applegpu/tree/28dd80468d4a6c94fa4dac34ed488196f5fedc5f/hwtestbed).
No upstream macOS harness code is vendored here; the carrier, bounded parser,
Objective-C interface and Python executor are independent implementations.

USC/CDM packet construction follows Mesa's `src/asahi/lib/cmdbuf.xml`, including
the independently reviewed Mesa 24.0.0 register flag and launch field definitions:

Copyright 2021–2025 Alyssa Rosenzweig.
Copyright 2023–2025 Valve Corporation.
SPDX-License-Identifier: MIT.

These definitions carry the MIT terms reproduced in the repository's
[LICENSE](../../LICENSE). Keep these notices with the derived packet implementation.
Mesa is a source reference, not a runtime dependency.

The residency list and trace-ID range layout were also checked against
Mesa 22.2.0 `src/asahi/lib/io.h` and `agx_device.c`, and
`src/gallium/drivers/asahi/magic.c` (Copyright 2021 Alyssa Rosenzweig;
Copyright 2019 Collabora, Ltd.; MIT). The notification ring boundary follows
Apple's open-source IOKitUser `IODataQueueClient.c`; no implementation is copied.

The AGXC v2 capture format and relocation approach were inspected in the user's
local applegpu experiments (`experimental/iokit_capture.c` and `cap_format.py`).
The freezer and executor here are independent implementations. The retained
IOGPU envelope comes from AGXForge's own measured base-M1 store dispatch on
build 26A434. Original shader bytes and userspace Metal object pointers are
scrubbed from the template. Apple frameworks and shaders are not redistributed.
`make macos-support` prepares a local driver support cache in the ignored build
directory using pipeline creation in a separate Metal process. Its interposer
refuses GPU submissions. Native execution restores that locally generated cache
and places compiled programs beyond it; cache hashes are retained in receipts.
The macOS SDK supplies the IOKit, IODataQueue and Metal interfaces at build time.

The separately downloaded Qwen checkpoint retains its upstream license and is
identified by the existing pinned revision and each model run's checkpoint hashes.
