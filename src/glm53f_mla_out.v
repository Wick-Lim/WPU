//============================================================================
// glm53f_mla_out.v -- the output side of GLM-5.3-Flash MLA, streamed off the
// shared glm_matmul_q4k with Q8_0 weights.
//
//     ctx[h] = W_uv[h] @ ctx_lat[h]        one pass per head   [V_DIM]
//     y      = W_o     @ ctx                                   [MODEL_DIM]
//
// WHY THE INPUT IS ctx_lat AND NOT ctx.  glm53f_mla_score returns the context in
// the LATENT basis, deliberately: W_uv is linear, so
//     SUM_j p_j (W_uv . ckv_j)  ==  W_uv . (SUM_j p_j ckv_j)
// and expanding ONCE here beats building a [H, V_DIM] value for every cached
// token. At the real shapes that is H*V*KVL = 64*256*512 = 8.4 M MACs once per
// token instead of once per CACHED token.
//
// IT IS NOT FREE IN fp, and the golden matches the FORM rather than the formula:
// measured in tools/glm53f_mla_ref.py, folding the value side moves 693 of 768
// outputs (rel 8.2e-05) at the checkpoint's widths against the expanded form.
// This unit implements the folded form, so the reference is called with
// absorb_v=True.
//
// Q8_0 needs nothing new from the engine ([scan]: the attention tensors are Q8_0):
// w_type = 2, code on w_hp[7:0], fp16 block scale on w_q8_d. The Q4_K header buses
// are tied off -- the default arm never reads them under Q8_0.
//
// TN AND THE TAIL: rows are processed TN at a time with no partial final group, so
// V_DIM and MODEL_DIM must both be multiples of TN. Checked at elaboration.
//============================================================================
`timescale 1ns/1ps
`ifndef GLM53F_MLA_OUT_V
`define GLM53F_MLA_OUT_V
`include "glm_fp.vh"

module glm53f_mla_out #(
    parameter integer MODEL_DIM = 16,
    parameter integer H         = 2,      // heads (real 64)
    parameter integer KVL       = 8,      // kv_lora_rank (real 512)
    parameter integer VD        = 8,      // v_head_dim (real 256)
    parameter integer TN        = 2,
    parameter integer KMAX      = 32,     // >= max(KVL, H*VD)
    parameter integer PMAX_OUT  = (VD > MODEL_DIM) ? VD : MODEL_DIM,
    parameter integer PMAX_IN   = (KVL > H*VD) ? KVL : H*VD
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    start,
    output reg                     busy,
    output reg                     done,

    input  wire [16*H*KVL-1:0]     ctx_lat_in,   // context in the LATENT basis

    output wire                    w_req,
    output wire                    w_sel,        // 0 = W_uv (per head), 1 = W_o
    output wire [$clog2(H)-1:0]    w_head,
    output wire [$clog2(PMAX_OUT/TN+1)-1:0] w_grp,
    output wire [$clog2(KMAX+1)-1:0]        w_k,
    input  wire [16*TN-1:0]        w_hp,
    input  wire [16*TN*((KMAX+31)/32)-1:0] w_q8_d,

    output reg  [16*MODEL_DIM-1:0] y_out
);
    localparam integer NSB = (KMAX + 255) / 256;
    localparam integer KW  = $clog2(KMAX + 1);
    localparam integer GW  = $clog2(PMAX_OUT/TN + 1);
    localparam integer HW  = (H <= 1) ? 1 : $clog2(H);

`ifndef YOSYS
    initial begin
        if ((VD % TN) || (MODEL_DIM % TN))
            $fatal(1, "glm53f_mla_out: V_DIM and MODEL_DIM must be multiples of TN (no partial final group)");
        if (KMAX < KVL || KMAX < H*VD)
            $fatal(1, "glm53f_mla_out: KMAX must cover both reductions (KV_LORA and H*V_DIM)");
    end
`endif

    reg           sel;
    reg [HW-1:0]  hd;
    reg [KW-1:0]  kcnt, klen;
    reg [GW-1:0]  grp, ngrp;
    reg [16*PMAX_IN-1:0]  in_r;
    reg [16*H*VD-1:0]     ctxv;          // the expanded context, bf16

    localparam [2:0] S_IDLE=3'd0, S_PREP=3'd1, S_STREAM=3'd2, S_WAIT=3'd3,
                     S_NEXT=3'd4, S_FIN=3'd5;
    reg [2:0] st;

    reg           mm_start;
    reg  [KW-1:0] mm_k_len;
    wire          mm_busy, mm_ov;
    wire [16*TN-1:0] mm_c;
    wire          stream = (st == S_STREAM);
    wire [3*TN-1:0] w_type_q8 = {TN{3'd2}};

    assign w_req  = stream;
    assign w_sel  = sel;
    assign w_head = hd;
    assign w_grp  = grp;
    assign w_k    = kcnt;
    wire [15:0] a_col = in_r[16*kcnt +: 16];

    glm_matmul_q4k #(.PE_M(1), .PE_N(TN), .KMAX(KMAX)) u_mm (
        .clk(clk), .rst(rst), .start(mm_start), .k_len(mm_k_len),
        .w_d({16*TN*NSB{1'b0}}), .w_dmin({16*TN*NSB{1'b0}}),
        .w_scales({96*TN*NSB{1'b0}}),
        .in_valid(stream), .a_col(a_col), .w_q({4*TN{1'b0}}),
        .busy(mm_busy), .out_valid(mm_ov), .c_out(mm_c),
        .w_type(w_type_q8), .w_hp(w_hp),
        .w_q6_sc({128*TN*NSB{1'b0}}), .w_q8_d(w_q8_d));

    integer gi;
    task setup_pass(input s_in, input integer rows, input integer kk);
        reg [31:0] rr, k32;
        begin
            k32 = kk; rr = rows / TN;
            klen <= k32[KW-1:0]; mm_k_len <= k32[KW-1:0];
            ngrp <= rr[GW-1:0]; grp <= 0;
            sel  <= s_in;
            mm_start <= 1'b1; st <= S_PREP;
        end
    endtask

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0; mm_start <= 1'b0;
            sel <= 1'b0; hd <= 0; grp <= 0; kcnt <= 0;
        end else begin
            done <= 1'b0; mm_start <= 1'b0;
            case (st)
                S_IDLE: if (start) begin
                    busy <= 1'b1; hd <= 0;
                    for (gi = 0; gi < KVL; gi = gi + 1)
                        in_r[16*gi +: 16] <= ctx_lat_in[16*gi +: 16];   // head 0
                    setup_pass(1'b0, VD, KVL);
                end

                S_PREP:   begin kcnt <= 0; st <= S_STREAM; end
                S_STREAM: begin
                    if (kcnt == klen - 1'b1) st <= S_WAIT;
                    kcnt <= kcnt + 1'b1;
                end

                S_WAIT: if (mm_ov) begin
                    for (gi = 0; gi < TN; gi = gi + 1) begin
                        if (!sel) ctxv[16*(hd*VD + grp*TN + gi) +: 16] <= mm_c[16*gi +: 16];
                        else      y_out[16*(grp*TN + gi) +: 16]        <= mm_c[16*gi +: 16];
                    end
                    if (grp == ngrp - 1'b1) st <= S_NEXT;
                    else begin grp <= grp + 1'b1; mm_start <= 1'b1; st <= S_PREP; end
                end

                S_NEXT: begin
                    if (!sel) begin
                        if (hd == H[HW-1:0] - 1'b1) begin
                            // every head expanded -> one W_o pass over the whole ctx
                            for (gi = 0; gi < H*VD; gi = gi + 1)
                                in_r[16*gi +: 16] <= ctxv[16*gi +: 16];
                            setup_pass(1'b1, MODEL_DIM, H*VD);
                        end else begin
                            for (gi = 0; gi < KVL; gi = gi + 1)
                                in_r[16*gi +: 16] <= ctx_lat_in[16*((hd+1'b1)*KVL + gi) +: 16];
                            hd <= hd + 1'b1;
                            setup_pass(1'b0, VD, KVL);
                        end
                    end else st <= S_FIN;
                end

                S_FIN: begin done <= 1'b1; busy <= 1'b0; st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
`endif // GLM53F_MLA_OUT_V
