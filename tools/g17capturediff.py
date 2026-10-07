#!/usr/bin/env python3
"""Compare two passive Submit captures of the authored A/B control (tools/g17capturerepeat.py's format), byte by byte.

    python3 tools/g17capturediff.py OLD.json NEW.json [--json OUT]

Each capture holds two runs. A byte that differs between the two runs of either capture is VOLATILE (an address, an
identifier, a timestamp, or the A/B's own immediate); a byte that is the same in both runs of each capture but differs
between the captures is a STABLE difference: what changed between the two platforms. Compared: the 64-byte Submit
record, every selector-9 allocation request, every mapped window and the shared-memory snapshots. CPU only.
"""
import json
import sys


def event(run):
    return [m["payload"] for m in run["messages"]
            if m.get("type") == "send" and m["payload"].get("kind") == "submit_enter"][0]


def blobs(run):
    e = event(run)
    out = {"submit_record": bytes.fromhex(e["record_hex"])}
    for i, r in enumerate(e["allocationRequests"]):
        out["alloc_request_%02d" % i] = bytes.fromhex(r.get("input_hex", ""))
    for w in e["windows"]:
        buf = bytearray(w["size"])
        for p in w["pages"]:
            b = bytes.fromhex(p["hex"])
            buf[p["offset"]:p["offset"] + len(b)] = b
        out["window_%#x" % int(w["aperture"])] = bytes(buf)
    for k, v in e["snapshot"]["shmems"].items():
        out["shmem_" + k] = bytes.fromhex(v.get("hex", ""))
    return out


def compare(old, new):
    O, N = [blobs(r) for r in old["runs"]], [blobs(r) for r in new["runs"]]
    rows = {}
    for key in sorted(set(O[0]) | set(N[0])):
        parts = [O[0].get(key, b""), O[1].get(key, b""), N[0].get(key, b""), N[1].get(key, b"")]
        n = min(map(len, parts))
        volatile = {i for i in range(n) if parts[0][i] != parts[1][i] or parts[2][i] != parts[3][i]}
        stable = [i for i in range(n) if i not in volatile and parts[0][i] != parts[2][i]]
        rows[key] = dict(bytes=[len(parts[0]), len(parts[2])], volatile=len(volatile), stable_differences=len(stable),
                         first_stable_offsets=[hex(i) for i in stable[:24]])
    changed = [k for k, v in rows.items() if v["stable_differences"] or v["bytes"][0] != v["bytes"][1]]
    return dict(compared=len(rows), with_stable_differences=len(changed), changed=changed, rows=rows)


def main(argv):
    rep = compare(json.load(open(argv[0])), json.load(open(argv[1])))
    if "--json" in argv:
        json.dump(rep, open(argv[argv.index("--json") + 1], "w"), indent=1)
    for k in rep["changed"]:
        r = rep["rows"][k]
        print("%-20s %6d stable differences  first %s" % (k, r["stable_differences"], " ".join(r["first_stable_offsets"][:8])))
    print("%d of %d structures differ between the captures beyond run-to-run variation"
          % (rep["with_stable_differences"], rep["compared"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
