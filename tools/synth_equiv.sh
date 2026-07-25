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

# Hierarchical modules need their submodules present.  Read every other src/*.v
# alongside the version of THIS module under test, so `prep -top` resolves the
# whole subtree (yosys only elaborates what the top actually instantiates).
OTHERS=""
for f in "$REPO"/src/*.v; do
    [ "$(basename "$f")" = "$MOD.v" ] && continue
    OTHERS="$OTHERS $f"
done

run() {  # $1 = the module-under-test source, $2 = output stat file
    yosys -q -p "read_verilog -sv -I $REPO/src $1 $OTHERS; ${CHP} prep -top $MOD -flatten; opt -full; opt_clean -purge; tee -o $2 stat" 2>/dev/null
}

run "$TMP/ref.v" "$TMP/ref.txt" || { echo "synth_equiv: yosys failed on the reference"; exit 2; }
run "$SRC"       "$TMP/new.txt" || { echo "synth_equiv: yosys failed on the working tree"; exit 2; }

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
