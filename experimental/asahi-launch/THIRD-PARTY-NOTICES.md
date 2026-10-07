# Asahi source references

The files in `agxforge/g13/_vendor/` come from Dougall Johnson's applegpu,
revision `28dd80468d4a6c94fa4dac34ed488196f5fedc5f`. They retain the
[BSD 3-Clause license](../../agxforge/g13/_vendor/LICENSE). Only package-relative
imports were changed. They implement the encoding/disassembly reference tables
and the partial emulator; no applegpu Metal hardware harness is included.

G13 USC/CDM packet construction follows Mesa's `src/asahi/lib/cmdbuf.xml`,
revision `e24dc5bd1e7fe6101bdc866fb16a15a8fcae1aae`:

Copyright 2021–2025 Alyssa Rosenzweig.
Copyright 2023–2025 Valve Corporation.
SPDX-License-Identifier: MIT.

These packet definitions carry the MIT permission and warranty terms reproduced
in the repository's [LICENSE](../../LICENSE). Keep these copyright notices with
redistributions of the derived packet implementation. Mesa helpers are not
vendored or dynamically linked. Additional Mesa files were reviewed as source
references; their exact hashes are recorded in `sources.json`.

The C helper includes the installed Linux DRM and Asahi UAPI headers. It does
not copy those headers into this repository. The tested kernel source revision
and header hash are recorded in `sources.json`. The Python wrapper crosses an
opaque C interface and does not duplicate kernel structure layouts.

The public Qwen checkpoint is downloaded separately, pinned by revision and
identified by hashes in each model run. No model weights are included in this
source tree or relicensed by this project.
