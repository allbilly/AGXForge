# Build and check the retained release on macOS. Explicit macos-* execution targets
# below dispatch base-M1 G13 programs; the retained release checks do not dispatch.
#
#   make native-tools   the decoder wrapper and the native harnesses (clang from Xcode's command-line tools)
#   make examples       workflows 1-4: compile (CPU), simulate (CPU), and two receipt checks (evidence only)
#   make test           the retained tests (release/tests.txt in the research checkout chose them)
#   make check          all three
#   make platform       compare this machine with the measured one (it does not fail on a difference; it says what it bears on)
PYTHON ?= python3
CC = clang
CXX = clang++
NATIVE_CFLAGS ?= -O2
EXAMPLE_OUT ?= $(shell mktemp -d -t agxforge-example.XXXXXX)/out

.PHONY: all native-tools examples test check platform
all: native-tools

native-tools: tools/agx3dis tools/agx3meta tools/libagx3dis.dylib tools/g17scanworker spike/accel/libaccel.dylib \
	tools/g17decodegen tools/g17specgen tools/g17twinrun

# Apple's G17 decoder, reached through GPUCompiler.framework at run time (the compiler's release checks use it)
tools/agx3dis: tools/agx3dis.c tools/agx3remap.h tools/agx3renumber.h
	$(CC) $(NATIVE_CFLAGS) -o $@ $<

tools/libagx3dis.dylib: tools/agx3dislib.c tools/agx3remap.h tools/agx3renumber.h
	$(CC) $(NATIVE_CFLAGS) -dynamiclib -o $@ $<

tools/agx3meta: tools/agx3meta.c tools/agx3renumber.h
	$(CC) $(NATIVE_CFLAGS) -o $@ $<

tools/g17scanworker: tools/g17scanworker.m tools/g17scanstorage.h
	$(CC) -fobjc-arc $(NATIVE_CFLAGS) -Wall -Wextra -Werror -framework Foundation -framework Metal -o $@ $<

# the chained-token executor through Metal (one command buffer a token; MM 25.138)
tools/g17decodegen: tools/g17decodegen.m tools/g17gpulock.h
	$(CC) -fobjc-arc $(NATIVE_CFLAGS) -I tools -framework Foundation -framework Metal -o $@ $<

tools/g17specgen: tools/g17specgen.m tools/g17gpulock.h
	$(CC) -fobjc-arc $(NATIVE_CFLAGS) -Wall -Wextra -Werror -I tools -framework Foundation -framework Metal -o $@ $<

# the matched study's kernel harness: a bundle and its Apple-compiled twin on the same buffers (MM 25.211)
tools/g17twinrun: tools/g17twinrun.m tools/g17gpulock.h
	$(CC) -fobjc-arc $(NATIVE_CFLAGS) -Wall -Wextra -Werror -I tools -framework Foundation -framework Metal -o $@ $<

spike/accel/libaccel.dylib: spike/accel/accel.mm tools/g17gpulock.h
	$(CXX) $(NATIVE_CFLAGS) -dynamiclib -fobjc-arc -framework Foundation -framework Metal $< -o $@

examples: native-tools
	$(PYTHON) examples/tensor_17x19x16.py $(EXAMPLE_OUT)
	$(PYTHON) examples/tensor_feed_rule.py
	$(PYTHON) examples/decode_study.py
	$(PYTHON) examples/native_inference.py

test: native-tools
	$(PYTHON) tools/release_tests.py

check: examples test

platform:
	$(PYTHON) tools/g17platform.py --check

# Independent Linux/G13 bring-up. Does not build Apple's G17 tools.
ASAHI_CC ?= cc
ASAHI_CFLAGS ?= -O2 -g -std=c11 -Wall -Wextra -Werror
ASAHI_SRC = agxforge/runtime/asahi.c agxforge/runtime/asahi_launch.c
ASAHI_HEADERS = agxforge/runtime/asahi.h agxforge/runtime/asahi_internal.h
.PHONY: asahi-tools asahi-probe asahi-smoke asahi-kernels asahi-isa g13-test
asahi-tools: build/asahi/libagxforge_asahi.so build/asahi/00_probe

build/asahi/libagxforge_asahi.so: $(ASAHI_SRC) $(ASAHI_HEADERS)
	mkdir -p build/asahi
	$(ASAHI_CC) $(ASAHI_CFLAGS) -fPIC -shared -o $@ $(ASAHI_SRC)

build/asahi/00_probe: examples/asahi/00_probe.c $(ASAHI_SRC) $(ASAHI_HEADERS)
	mkdir -p build/asahi
	$(ASAHI_CC) $(ASAHI_CFLAGS) -Iagxforge/runtime -o $@ $< $(ASAHI_SRC)

asahi-probe: asahi-tools
	build/asahi/00_probe

asahi-smoke: asahi-tools
	$(PYTHON) examples/asahi/smoke.py --output results/asahi-smoke

g13-test:
	$(PYTHON) -m unittest discover -s test/g13 -v

asahi-kernels: asahi-tools
	$(PYTHON) examples/asahi/verify_kernels.py --output results/asahi-kernels

asahi-isa: asahi-tools
	$(PYTHON) examples/asahi/verify_isa.py --output results/asahi-isa

# Base M1/G13G machine code loaded through a local Metal carrier archive.
MACOS_BACKEND ?= metal
MACOS_COMPILER ?= g13
.PHONY: macos-tools macos-support macos-probe macos-smoke macos-isa macos-kernels macos-test macos-mesa-tools
macos-tools: build/macos/libagxforge_macos.dylib build/macos/libagxforge_iogpu.dylib

build/macos/libagxforge_macos.dylib: agxforge/runtime/macos.m
	mkdir -p build/macos
	xcrun clang -O2 -fobjc-arc -Wall -Wextra -Werror -dynamiclib -framework Foundation -framework Metal -o $@ $<

build/macos/libagxforge_iogpu.dylib: agxforge/runtime/iogpu.c
	mkdir -p build/macos
	xcrun clang -O2 -std=c11 -Wall -Wextra -Werror -dynamiclib -framework IOKit -o $@ $<

build/macos/libagxforge_support.dylib: tools/macos_support.c
	mkdir -p build/macos
	xcrun clang -O2 -std=c11 -Wall -Wextra -Werror -dynamiclib -framework IOKit -o $@ $<

macos-support: macos-tools build/macos/libagxforge_support.dylib
	$(PYTHON) tools/macos_support.py

macos-probe: macos-tools
	$(PYTHON) examples/macos/probe.py --backend $(MACOS_BACKEND)

macos-smoke: macos-tools
	$(PYTHON) examples/macos/smoke.py --backend $(MACOS_BACKEND) --output results/macos-$(MACOS_BACKEND)-smoke

macos-isa: macos-tools
	$(PYTHON) examples/macos/verify_isa.py --backend $(MACOS_BACKEND) --output results/macos-$(MACOS_BACKEND)-isa

macos-kernels: macos-tools
	$(PYTHON) examples/macos/verify_kernels.py --backend $(MACOS_BACKEND) --compiler $(MACOS_COMPILER) --output results/macos-$(MACOS_BACKEND)-$(MACOS_COMPILER)-kernels

macos-mesa-tools:
	$(PYTHON) tools/build_mesa_agx.py

macos-test: macos-tools g13-test
