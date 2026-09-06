//============================================================================
// glm53f_kda_gemv.v -- the nine KDA projections, streamed off the shared
// glm_matmul_q4k with Q8_0 weights.
//
// This is what turns glm53f_kda_layer from a unit into a layer: the projections
// stop being handed to it and start being FETCHED. It implements the same
// proj_req / proj_sel / proj_in / proj_done / proj_out service the layer already
// talks to, so the layer is unchanged and its gate stays valid.
//
// Q8_0 IS ALREADY AN ENGINE FEATURE, and that is the whole point.  Every KDA
// projection is Q8_0 in the GGUF (attn_{q,k,v,output}, ssm_{beta,f_a,f_b,g_a,g_b}
// [scan]). glm_matmul_q4k takes `w_type` (2 = Q8_0), the code on `w_hp[7:0]` and
// the fp16 block scale on `w_q8_d`; weight_loader_q4k already emits those lanes.
// Nothing shared had to change -- docs/GLM53_FLASH_PORT.md 4.3g used to say
// otherwise and was corrected. The Q4_K header buses (w_q/w_d/w_dmin/w_scales) are
// tied off: they are the DEFAULT arm's inputs and Q8_0 never reads them.
//
// PROTOCOL, copied from swiglu_expert_q4k rather than invented: raise w_req while
// streaming, publish (w_sel, w_grp, w_k) so the system can answer combinationally
// with this beat's codes, present the group's scales, pulse `start` with `k_len`,
// stream k_len beats of (a_col, w_hp), then wait for out_valid.  One output group
// of TN columns per pass.
//
// bf16 IN AND OUT, deliberately.  a_col is bf16 and c_out is bf16 -- the model's
// own linear layers are bf16-out and every activation in this repo is bf16, so
// this is faithful, not a concession. The layer's proj_out port is fp32-TYPED and
// carries bf16-VALUED data, which is why dropping this engine in front of it
// needs no change to the layer.
//
// TN AND THE TAIL.  Output rows are processed TN at a time and this module does
// NOT handle a partial final group -- every projection's row count must be a
// multiple of TN. That is checked at elaboration rather than assumed. At the real
// shape the row counts are 8192 / 4096 / 128 / 64, so the smallest is 64 and TN up
// to 64 is legal; on the H=2 slice `ssm_beta` has only H=2 rows, which is why the
// slice runs TN=2.
//============================================================================
`timescale 1ns/1ps
`ifndef GLM53F_KDA_GEMV_V
`define GLM53F_KDA_GEMV_V
`include "glm_fp.vh"

module glm53f_kda_gemv #(
    parameter integer MODEL_DIM = 16,
    parameter integer H         = 2,
    parameter integer DK        = 4,
    parameter integer DV        = 4,
    parameter integer RANK      = 4,
    parameter integer TN        = 2,     // output columns per pass (= matmul PE_N)
    parameter integer KMAX      = 32,    // >= max projection K
    parameter integer PMAX_IN   = (MODEL_DIM > H*DV) ? MODEL_DIM : H*DV,
    parameter integer PMAX_OUT  = (H*DK > MODEL_DIM) ? H*DK : MODEL_DIM
)(
    input  wire                    clk,
    input  wire                    rst,

    // ---- the service the layer consumes ----
    input  wire                    proj_req,
    input  wire [3:0]              proj_sel,
    input  wire [32*PMAX_IN-1:0]   proj_in,
    output reg                     proj_done,
    output reg  [32*PMAX_OUT-1:0]  proj_out,

    // ---- the weight pull this raises at the system ----
    output wire                    w_req,
    output wire [3:0]              w_sel,
    output wire [$clog2(PMAX_OUT/TN+1)-1:0] w_grp,
    output wire [$clog2(KMAX+1)-1:0]        w_k,
    input  wire [16*TN-1:0]        w_hp,      // Q8_0 code in [7:0] per column
    input  wire [16*TN*((KMAX+31)/32)-1:0] w_q8_d   // fp16 d per (col, 32-block)
);
    localparam integer NSB = (KMAX + 255) / 256;
    localparam integer KW  = $clog2(KMAX + 1);
    localparam integer GW  = $clog2(PMAX_OUT/TN + 1);

    // per-projection shape
    function integer prows(input [3:0] s);
        begin
            case (s)
                4'd0, 4'd1: prows = H*DK;
                4'd2:       prows = H*DV;
                4'd3:       prows = H;
                4'd4, 4'd6: prows = RANK;
                4'd5:       prows = H*DK;
                4'd7:       prows = H*DV;
                default:    prows = MODEL_DIM;
            endcase
        end
    endfunction
    function integer pcols(input [3:0] s);
        begin
            case (s)
                4'd5, 4'd7: pcols = RANK;
                4'd8:       pcols = H*DV;
                default:    pcols = MODEL_DIM;
            endcase
        end
    endfunction

`ifndef YOSYS
    integer chk;
    initial begin
        for (chk = 0; chk <= 8; chk = chk + 1) begin
            if (prows(chk[3:0]) % TN != 0)
                $fatal(1, "glm53f_kda_gemv: projection row count is not a multiple of TN; this module has no partial-group tail path");
            if (pcols(chk[3:0]) > KMAX)
                $fatal(1, "glm53f_kda_gemv: KMAX is smaller than a projection's K");
        end
    end
`endif

    reg  [3:0]  sel_r;
    reg  [KW-1:0] kcnt, klen;
    reg  [GW-1:0] grp, ngrp;
    reg  [32*PMAX_IN-1:0] in_r;

    localparam [2:0] S_IDLE=3'd0, S_PREP=3'd1, S_STREAM=3'd2, S_WAIT=3'd3, S_FIN=3'd4;
    reg [2:0] st;

    reg           mm_start;
    reg  [KW-1:0] mm_k_len;
    wire          mm_busy, mm_ov;
    wire [16*TN-1:0] mm_c;

    wire stream = (st == S_STREAM);
    assign w_req = stream;
    assign w_sel = sel_r;
    assign w_grp = grp;
    assign w_k   = kcnt;

    // one bf16 activation per beat (PE_M = 1)
    wire [15:0] a_col = fp32_to_bf16(in_r[32*kcnt +: 32]);

    // Q8_0 for every lane; the Q4_K header buses are the default arm's and unused.
    // ---- must-fail injections; none defined in a normal build ----
`ifdef INJ_KGV_Q4K_TYPE
    // must FAIL: w_type left at Q4_K. This is the exact silent-wrong-weights case
    // the engine's own header warns about -- an UNDRIVEN w_type also reads as
    // Q4_K, so "forgot to drive it" and "drove it wrong" look identical.
    wire [3*TN-1:0] w_type_q8 = {TN{3'd0}};
`else
    wire [3*TN-1:0] w_type_q8 = {TN{3'd2}};
`endif

    glm_matmul_q4k #(.PE_M(1), .PE_N(TN), .KMAX(KMAX)) u_mm (
        .clk(clk), .rst(rst), .start(mm_start), .k_len(mm_k_len),
        .w_d({16*TN*NSB{1'b0}}), .w_dmin({16*TN*NSB{1'b0}}),
        .w_scales({96*TN*NSB{1'b0}}),
        .in_valid(stream), .a_col(a_col), .w_q({4*TN{1'b0}}),
        .busy(mm_busy), .out_valid(mm_ov), .c_out(mm_c),
        .w_type(w_type_q8), .w_hp(w_hp),
`ifdef INJ_KGV_NO_SCALE
        // must FAIL: the fp16 block scales dropped, so every Q8_0 code dequants
        // against d = 0 and the whole projection collapses.
        .w_q6_sc({128*TN*NSB{1'b0}}), .w_q8_d({16*TN*((KMAX+31)/32){1'b0}})
`else
        .w_q6_sc({128*TN*NSB{1'b0}}), .w_q8_d(w_q8_d)
`endif
    );

    integer gi;
    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; proj_done <= 1'b0; mm_start <= 1'b0;
            grp <= 0; kcnt <= 0;
        end else begin
            proj_done <= 1'b0; mm_start <= 1'b0;
            case (st)
                S_IDLE: if (proj_req) begin
                    sel_r <= proj_sel; in_r <= proj_in;
                    klen  <= pcols(proj_sel);
                    ngrp  <= prows(proj_sel) / TN;
                    mm_k_len <= pcols(proj_sel);
                    grp <= 0; mm_start <= 1'b1; st <= S_PREP;
                end
                S_PREP: begin kcnt <= 0; st <= S_STREAM; end
                S_STREAM: begin
                    if (kcnt == klen - 1'b1) st <= S_WAIT;
                    kcnt <= kcnt + 1'b1;
                end
                S_WAIT: if (mm_ov) begin
                    for (gi = 0; gi < TN; gi = gi + 1)
`ifdef INJ_KGV_GRP_ALIAS
                        // must FAIL: every output group written to slot 0, so only
                        // the LAST group survives and the rest of the vector is
                        // stale. Deliberately still TERMINATES -- an injection that
                        // hangs the FSM only fails by timeout, and would cost the
                        // release gate 3M simulated cycles on every run.
                        proj_out[32*(0*TN + gi) +: 32] <= bf16_to_fp32(mm_c[16*gi +: 16]);
`else
                        proj_out[32*(grp*TN + gi) +: 32] <= bf16_to_fp32(mm_c[16*gi +: 16]);
`endif
                    if (grp == ngrp - 1'b1) st <= S_FIN;
                    else begin
                        grp <= grp + 1'b1;
                        mm_start <= 1'b1; st <= S_PREP;
                    end
                end
                S_FIN: begin proj_done <= 1'b1; st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
`endif // GLM53F_KDA_GEMV_V
