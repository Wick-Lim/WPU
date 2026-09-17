//============================================================================
// glm53f_wdesc.v -- the per-tensor weight descriptor GLM-5.3-Flash needs and
// this system has never had for ANY type.
//
//     (kind, layer, expert)  ->  (base, klen, nsblk, wtype)
//
// WHY IT IS NEEDED NOW.  glm_q4k_system drives weight_loader_q4k with a HARDCODED
// single tile -- desc_base = 0, desc_nsblk = 1 -- and leaves desc_wtype UNDRIVEN,
// which the loader reads as Q4_K. That was survivable while one model, one type
// and one tile were in play. It is not survivable here: [scan] says
// ffn_{gate,up}_exps is Q4_K x42 + Q5_K x1 and ffn_down_exps is Q5_K x40 + Q6_K
// x3, so the TYPE varies per tensor AND per layer, and a model top that walks 45
// layers has to address a different tile for every (layer, expert).
//
// ---- SHAPE: A PER-KIND TABLE PLUS A SHORT EXCEPTION LIST ----
// A flat descriptor for every tensor would be 1412 entries. It does not need to
// be, because the checkpoint is laid out regularly: within a KIND, layers are a
// fixed stride apart and experts within a tensor are uniform. So a kind carries
// (base, layer stride, expert stride, klen, nsblk, default type) -- about twenty
// entries -- and the irregularity that is left is exactly the UD quantisation
// bumps, which are a handful of (kind, layer) -> type exceptions.
//
// THAT SPLIT IS THE CHECKPOINT'S OWN, not a compression trick: "UD bump on
// blk.{11,12,44}.ffn_down_exps" is how the census describes it, and an exception
// list is what that sentence is. Encoding the bumps as exceptions rather than
// flattening the table keeps the regular part checkable by arithmetic and leaves
// the irregular part short enough to read.
//
// ---- WHAT A WRONG ANSWER LOOKS LIKE ----
// Both failure modes are silent, which is why both get a must-fail leg:
//   * miss an exception and the tile streams with Q5_K geometry where the bytes
//     are Q6_K -- same widths, wrong decode, no error;
//   * drop the expert stride and every expert reads expert 0's bytes -- a
//     perfectly well-formed model that has one expert 288 times.
//
// Purely combinational: the loader latches the descriptor on its own `start`.
//============================================================================
`timescale 1ns/1ps
`ifndef GLM53F_WDESC_V
`define GLM53F_WDESC_V

module glm53f_wdesc #(
    parameter integer NKIND  = 8,
    parameter integer NLAYER = 4,
    parameter integer NEXC   = 2,      // (kind, layer) -> wtype exceptions
    parameter integer NEXP   = 4,
    parameter integer ADDR_W = 32,
    parameter integer KW     = 16,
    parameter integer SBW    = 8,
    parameter integer KIDW   = (NKIND  <= 1) ? 1 : $clog2(NKIND),
    parameter integer LIDW   = (NLAYER <= 1) ? 1 : $clog2(NLAYER),
    parameter integer EIDW   = (NEXP   <= 1) ? 1 : $clog2(NEXP),
    // per-kind table, packed low-index-first
    parameter [NKIND*ADDR_W-1:0] K_BASE  = {NKIND*ADDR_W{1'b0}},  // (layer 0, expert 0)
    parameter [NKIND*ADDR_W-1:0] K_LSTR  = {NKIND*ADDR_W{1'b0}},  // per-layer stride
    parameter [NKIND*ADDR_W-1:0] K_ESTR  = {NKIND*ADDR_W{1'b0}},  // per-expert stride
    parameter [NKIND*KW-1:0]     K_KLEN  = {NKIND*KW{1'b0}},
    parameter [NKIND*SBW-1:0]    K_NSBLK = {NKIND*SBW{1'b0}},
    parameter [NKIND*3-1:0]      K_WTYPE = {NKIND*3{1'b0}},
    // the UD bumps
    parameter [NEXC*KIDW-1:0]    X_KIND  = {NEXC*KIDW{1'b0}},
    parameter [NEXC*LIDW-1:0]    X_LAYER = {NEXC*LIDW{1'b0}},
    parameter [NEXC*3-1:0]       X_WTYPE = {NEXC*3{1'b0}},
    parameter [NEXC-1:0]         X_VALID = {NEXC{1'b0}}
)(
    input  wire [KIDW-1:0]   kind,
    input  wire [LIDW-1:0]   layer,
    input  wire [EIDW-1:0]   expert,
    output wire [ADDR_W-1:0] desc_base,
    output wire [KW-1:0]     desc_klen,
    output wire [SBW-1:0]    desc_nsblk,
    output wire [2:0]        desc_wtype
);
    wire [ADDR_W-1:0] kb = K_BASE[kind*ADDR_W +: ADDR_W];
    wire [ADDR_W-1:0] ls = K_LSTR[kind*ADDR_W +: ADDR_W];
    wire [ADDR_W-1:0] es = K_ESTR[kind*ADDR_W +: ADDR_W];

`ifdef INJ_WDESC_NO_ESTR
    // must FAIL: drop the expert stride, so every expert reads expert 0's bytes.
    // The result is a well-formed model that has one expert 288 times.
    assign desc_base = kb + ls * layer;
`else
    assign desc_base = kb + ls * layer + es * expert;
`endif
    assign desc_klen  = K_KLEN[kind*KW +: KW];
    assign desc_nsblk = K_NSBLK[kind*SBW +: SBW];

    // type = the kind's default, overridden by a matching (kind, layer) exception
    reg [2:0] wt;
    integer   e;
    always @* begin
        wt = K_WTYPE[kind*3 +: 3];
`ifndef INJ_WDESC_NO_EXC
        // must FAIL when defined: ignore the UD bumps, so blk.{11,12,44}
        // ffn_down_exps streams Q5_K geometry over Q6_K bytes -- same widths,
        // wrong decode, no error anywhere.
        for (e = 0; e < NEXC; e = e + 1)
            if (X_VALID[e]
                && (X_KIND [e*KIDW +: KIDW] == kind)
                && (X_LAYER[e*LIDW +: LIDW] == layer))
                wt = X_WTYPE[e*3 +: 3];
`endif
    end
    assign desc_wtype = wt;
endmodule
`endif // GLM53F_WDESC_V
