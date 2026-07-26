#!/usr/bin/env bash
# synth_equiv.sh -- per-module synthesis diff: "did this edit change the netlist?"
#
# WHY
#   The perf work in docs/ULTRA_PERF_MODULES.md is a long series of edits to
#   individual leaves, and the rule for every one of them is the same: the
#   DEFAULT build must stay exactly what it was.  Synthesizing the whole product
#   top to check that costs many minutes per edit (it timed out at 5 min during
#   rank 7), which is far too slow to run per change -- so the check gets skipped,
#   which is how "byte-identical default" claims quietly stop being true.
#
#   This does it at MODULE granularity instead: synthesize ONE module from a
#   reference git revision, synthesize the SAME module from the working tree, and
#   diff the cell histograms.  Small parameter overrides keep it to seconds.
#
# USAGE
#   tools/synth_equiv.sh <module> [rev] [chparam-overrides...]
#     module   : module name, expected at src/<module>.v
#     rev      : git revision to compare against (default HEAD)
#     overrides: e.g. "PE_M 2" "PE_N 2" "KMAX 256"  -- applied to BOTH sides, so
#                the comparison stays apples-to-apples while staying fast.
#
#   Exit 0 = the netlists are identical (the default path is untouched).
#   Exit 1 = they differ; the diff is printed.
#
# EXAMPLES
#   tools/synth_equiv.sh glm_matmul_q4k HEAD~1 "PE_M 2" "PE_N 2"
#   tools/synth_equiv.sh cdc_async_fifo            # vs HEAD, default params
#
# NOTE: this compares the SYNTHESIZED CELL HISTOGRAM, not a formal equivalence.
#   It catches "this edit changed the default hardware"; it does not by itself
#   prove two different netlists compute the same function.  For that the repo
#   uses its functional goldens (and yosys equiv_* where a proof is warranted).
set -u

MOD="${1:?usage: synth_equiv.sh <module> [rev] [\"PARAM VALUE\"...]}"
REV="${2:-HEAD}"
shift 2 2>/dev/null || shift 1 2>/dev/null || true

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/src/$MOD.v"
[ -f "$SRC" ] || { echo "synth_equiv: no such module source: src/$MOD.v"; exit 2; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# reference copy of just this module
if ! git -C "$REPO" show "$REV:src/$MOD.v" > "$TMP/ref.v" 2>/dev/null; then
    echo "synth_equiv: cannot read src/$MOD.v at revision $REV"; exit 2
fi

# chparam string shared by both sides.  NOTE: do NOT use `set --` here -- it
# clobbers the very "$@" being iterated.
CHP=""
for ov in "$@"; do
    pname="${ov%% *}"; pval="${ov##* }"
    [ -n "$pname" ] && [ "$pname" != "$ov" ] && CHP="$CHP -set $pname $pval"
done
[ -n "$CHP" ] && CHP="chparam$CHP $MOD;"

# SE_NEW_PARAMS -- overrides applied to the WORKING-TREE side ONLY.
#
#   The positional overrides above go to BOTH sides, which is right for keeping the
#   comparison apples-to-apples but makes one necessary check impossible: proving a
#   NEW parameter is live.  You cannot ask "old != new@P=1" when the old revision has
#   no P at all -- chparam just errors.  Without that, a gate can only ever report
#   IDENTICAL, and a check that cannot fail constrains nothing (repo rule 3).
#   So: run once with no SE_NEW_PARAMS (must be IDENTICAL -> the default is untouched),
#   then again with SE_NEW_PARAMS="P 1" (must DIFFER -> the parameter really builds
#   different hardware, so the first run was not vacuous).
CHP_NEW="$CHP"
if [ -n "${SE_NEW_PARAMS:-}" ]; then
    _n=""
    for ov in $SE_NEW_PARAMS; do _n="$_n $ov"; done
    set -- $_n
    _extra=""
    while [ $# -ge 2 ]; do _extra="$_extra -set $1 $2"; shift 2; done
    [ -n "$_extra" ] && CHP_NEW="$CHP chparam$_extra $MOD;"
fi

# Hierarchical modules need their submodules present -- but ONLY those.
#
# Reading every other src/*.v (what this did originally) is not merely slow, it is
# BROKEN: src/glm_fp.vh is `ifndef`-guarded and several modules `include` it INSIDE
# their module body, so Verilog-2001 binds those functions to whichever module is
# parsed first and every later file that needs them fails to resolve.  Concretely,
# `synth_equiv.sh swiglu_expert_q4k` died on the UNMODIFIED tree with
# "src/dsa_indexer.v:436: ERROR: Can't resolve function name bf16_to_fp32" -- i.e.
# the tool reported a failure that had nothing to do with the edit under test, on a
# module central to the biggest remaining perf item.
#
# So: walk the actual instantiation closure from $MOD and read only that.  Fewer
# files means the guard is consumed by a module that genuinely needs it, and the
# synth is faster besides.
declare_closure() {
    # name:file for EVERY module in every file -- a file may define many (glm_fp_pipe.v
    # defines the whole fp32 pipe family).  Taking only the first module per file was a
    # bug: the closure stopped one level down and the synth then failed for reasons
    # unrelated to the edit under test.
    _names=$(awk '/^[[:space:]]*module[[:space:]]/ {
                      n=$2; sub(/[#(].*/,"",n); if (n!="") print n ":" FILENAME
                  }' "$REPO"/src/*.v)
    _seen=" $MOD "
    _frontier="$MOD"
    while [ -n "$_frontier" ]; do
        _next=""
        for _cur in $_frontier; do
            for _pair in $_names; do
                [ "${_pair%%:*}" = "$_cur" ] || continue
                _curf="${_pair#*:}"
                for _p2 in $_names; do
                    _n="${_p2%%:*}"
                    case "$_seen" in *" $_n "*) continue ;; esac
                    if grep -qE "^[[:space:]]*${_n}[[:space:]]*#\\(|^[[:space:]]*${_n}[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\\(" "$_curf"; then
                        _seen="$_seen$_n "
                        _next="$_next $_n"
                    fi
                done
            done
        done
        _frontier="$_next"
    done
    # dedupe to FILES (many module names map to one file)
    OTHERS=""
    for _n in $_seen; do
        [ "$_n" = "$MOD" ] && continue
        for _p2 in $_names; do
            [ "${_p2%%:*}" = "$_n" ] || continue
            _f="${_p2#*:}"
            case " $OTHERS " in *" $_f "*) ;; *) OTHERS="$OTHERS $_f" ;; esac
        done
    done
}
declare_closure

run() {
    [ -n "${SE_DEBUG:-}" ] && echo "CLOSURE: $OTHERS" >&2  # $1 = the module-under-test source, $2 = output stat file
    # NOTE the read order: $OTHERS (the instantiation closure) BEFORE $1.  glm_fp.vh
    # is `ifndef`-guarded and included inside module bodies, so its functions bind to
    # whichever module parses FIRST; putting the module under test first starved its
    # own children and made this tool fail on unmodified sources.
    yosys -q -p "read_verilog -sv -I $REPO/src $OTHERS $1; $3 prep -top $MOD -flatten; opt -full; opt_clean -purge; tee -o $2 stat" 2>/dev/null
}

run "$TMP/ref.v" "$TMP/ref.txt" "$CHP" || { echo "synth_equiv: yosys failed on the reference"; exit 2; }
run "$SRC"       "$TMP/new.txt" "$CHP_NEW" || { echo "synth_equiv: yosys failed on the working tree"; exit 2; }

sed -n '/[0-9] cells$/,$p' "$TMP/ref.txt" > "$TMP/ref_cells.txt"
sed -n '/[0-9] cells$/,$p' "$TMP/new.txt" > "$TMP/new_cells.txt"

# a vacuous pass (both extractions empty) would be worse than useless
if [ ! -s "$TMP/ref_cells.txt" ] || [ ! -s "$TMP/new_cells.txt" ]; then
    echo "synth_equiv: FAILED -- the stat extraction is EMPTY (yosys output format changed); the comparison would be vacuous"
    exit 2
fi

if diff -q "$TMP/ref_cells.txt" "$TMP/new_cells.txt" >/dev/null; then
    echo "synth_equiv: $MOD IDENTICAL vs $REV  ($(head -1 "$TMP/ref_cells.txt" | tr -s ' ' | sed 's/^ *//'))"
    exit 0
else
    echo "synth_equiv: $MOD DIFFERS vs $REV"
    diff "$TMP/ref_cells.txt" "$TMP/new_cells.txt"
    exit 1
fi
