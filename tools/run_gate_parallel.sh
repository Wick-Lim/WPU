#!/usr/bin/env bash
# run_gate_parallel.sh -- run the release gate's targets in parallel, safely.
#
# WHY A DRIVER AND NOT JUST `make -j`.  Two reasons, both measured:
#
#  1. OUTPUT INTERLEAVING BREAKS THE VERDICT.  tools/check_test_counts.sh matches
#     `] ALL <n> TESTS PASSED` on ONE line.  Under `make -j` two targets write the
#     banner prefix and the vvp line as separate writes, so lines tear:
#         [q4k_prim] [glm_matmul_q4k
#         ALL 8 TESTS PASSED
#     and the checker then reports those gates MISSING. GNU make's --output-sync
#     fixes this, but it needs make 4.0+ and this machine has 3.81 (Xcode's), with
#     no 4.x installed. So each target gets its OWN make and its OWN log here, and
#     the logs are concatenated in the Makefile's declared order afterwards.
#
#  2. The shared generated inputs must exist BEFORE the fan-out. The Makefile now
#     declares them as real file targets built once (see "SHARED GENERATED INPUTS"
#     there), but independent `make` processes do not coordinate with each other --
#     only within one make. Building them here, serially, first, is what makes the
#     per-target makes safe.
#
# The file-level safety this depends on was established statically over every
# release-gate target: zero shared WRITE paths, zero cross-target read-after-write.
# Re-run that analysis if you add a gate that generates anything.
set -u

# HOLD THE MACHINE AWAKE.  This is not a nicety: two earlier full-gate runs were
# measured at 18.4 h and 49.0 h wall, of which 16.4 h and 41.2 h were the iMac in
# "Maintenance Sleep" (pmset -g log). Awake, those runs were 2.0 h and 7.8 h --
# i.e. normal. Sleep suspends the processes, so elapsed time accrues while CPU
# time does not, and the usual "is it hung?" test (CPU ~= elapsed) reports a
# HEALTHY process the whole time. Without this line a long gate silently takes
# days and every timing number derived from it is fiction.
# `caffeinate -w $$` holds the assertion for exactly this script's lifetime and
# exits with it. An earlier version re-exec'd the script under caffeinate instead;
# that left caffeinate running as a CHILD with no child of its own -- the
# assertion happened to be held, but by accident rather than by construction, and
# `ps` could not show the gate running under it. This form is verifiable: the
# caffeinate process names the pid it is waiting on.
CAFF_PID=""
if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i -s -w $$ &
    CAFF_PID=$!
    # DISOWN IT. Two reasons, and the first one deadlocks: the final bare `wait`
    # would otherwise wait on this job too, while caffeinate is waiting on THIS
    # shell -- measured, the driver hung after printing every PASS. The second is
    # that `jobs -rp` feeds the concurrency throttle, so an undisowned caffeinate
    # silently costs one of the JOBS slots.
    disown "$CAFF_PID" 2>/dev/null || true
    trap 'if [ -n "$CAFF_PID" ]; then kill "$CAFF_PID" 2>/dev/null; fi' EXIT
    echo "== holding the machine awake: caffeinate -i -s -w $$ (pid $CAFF_PID) =="
else
    echo "== WARNING: no caffeinate; a long run may be suspended by system sleep =="
fi

JOBS="${1:-6}"
MAKE="${MAKE:-/Applications/Xcode.app/Contents/Developer/usr/bin/make}"
LOGDIR=build/gatelogs
mkdir -p "$LOGDIR"

# The target list comes from the Makefile's own release-gate line -- one source of
# truth, so adding a gate there is enough. GATE_TARGETS overrides it for testing
# the driver itself on a cheap subset; the count check will then report every
# other gate MISSING, which is correct and expected.
TARGETS="${GATE_TARGETS:-$(awk '/^release-gate:/{sub(/^release-gate:/,""); print; exit}' Makefile)}"
[ -n "$TARGETS" ] || { echo "run_gate_parallel: cannot read the release-gate target list"; exit 1; }

# Long poles FIRST, so the critical path starts at t=0 instead of being scheduled
# last. Measured 2026-09-12..14 on the serial run: spec-slow 11.9 h, spec-greedy
# 10.7 h, expert-cache 8.6 h, intra-batch-verify 8.5 h. The gate cannot finish
# faster than its longest single target, so spec-slow sets the floor (~12 h).
LONG_FIRST="spec-slow spec-greedy expert-cache intra-batch-verify batched-q4k model-q4k model-q4k-acthw spec-adapt"

ORDERED=""
for t in $LONG_FIRST; do
    case " $TARGETS " in *" $t "*) ORDERED="$ORDERED $t";; esac
done
for t in $TARGETS; do
    case " $LONG_FIRST " in *" $t "*) ;; *) ORDERED="$ORDERED $t";; esac
done

echo "== prebuilding shared generated inputs (serial, once) =="
$MAKE gate-shared || { echo "FAILED: gate-shared"; exit 1; }

echo "== running $(echo $TARGETS | wc -w | tr -d ' ') targets, $JOBS at a time =="
START=$(date +%s)
rm -f "$LOGDIR"/*.status "$LOGDIR"/*.log 2>/dev/null
for t in $ORDERED; do
    while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$JOBS" ]; do sleep 2; done
    (
        s=$(date +%s)
        if $MAKE "$t" >"$LOGDIR/$t.log" 2>&1; then r=PASS; else r=FAIL; fi
        e=$(( $(date +%s) - s ))
        echo "$r $e" >"$LOGDIR/$t.status"
        printf '%-6s %-26s %5d s\n' "$r" "$t" "$e"
    ) &
done
wait
ELAPSED=$(( $(date +%s) - START ))

# concatenate in the Makefile's DECLARED order so the log reads like a serial run
: >build/release_gate.log
for t in $TARGETS; do
    [ -f "$LOGDIR/$t.log" ] && cat "$LOGDIR/$t.log" >>build/release_gate.log
done

FAILED=""
for t in $TARGETS; do
    st=$(awk '{print $1}' "$LOGDIR/$t.status" 2>/dev/null)
    [ "$st" = PASS ] || FAILED="$FAILED $t"
done
echo
printf '== wall %d s (%.1f h), %s at a time ==\n' "$ELAPSED" "$(echo "$ELAPSED/3600" | bc -l)" "$JOBS"
if [ -n "$FAILED" ]; then
    echo "FAILED targets:$FAILED"
    echo "  per-target logs are in $LOGDIR/"
    exit 1
fi
echo "release-gate: ALL gates passed (parallel)"
bash tools/check_test_counts.sh build/release_gate.log
