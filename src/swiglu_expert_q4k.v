`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// swiglu_expert_q4k.v -- GLM-5.2 SwiGLU FFN expert in Q4_K numerics.
//   y = ( silu(x @ W_gate) (.) (x @ W_up) ) @ W_down
//   with W_gate/W_up/W_down as GGML Q4_K-typed weights, NO re-quantization.
//   A Q4_K sibling of the prior swiglu_expert_fp8 (tag 'fp8-verified-baseline'): SAME FSM
//   (gate pass -> up pass -> silu*up merge -> down pass on ONE shared
//   glm_matmul_q4k), but the prior FP8 activation-shift machinery is GONE
//   (glm_matmul_q4k takes bf16 activations
//   directly), and the weight interface carries Q4_K codes + super-block scales.
//
// Verified bit-exact vs tools/q4k_ref.py (dequant + fp32 MAC) + glm_act silu.
//----------------------------------------------------------------------------
// WEIGHT INTERFACE (to the surrounding system / DMA buffer)
//   Per beat the module requests weight codes (w_req/w_sel/w_grp/w_k); the system
//   answers combinationally with 4-bit Q4_K codes for the active TN output columns
//   (w_q for gate|down, w_q_up for the up companion). The per-(column,super-block)
//   Q4_K scales for the current group are presented on w_d*/w_dmin*/w_scales* and
//   LATCHED by the matmul at its start. w_sel selects gate vs down weights.
//============================================================================
module swiglu_expert_q4k #(
    parameter integer HIDDEN = 128,   // model hidden size (scales to 6144)
    parameter integer INTER  = 64,    // FFN inter size (MoE 2048 / dense 12288)
    parameter integer TN     = 4,     // output-tile width = matmul PE_N
    parameter integer KMAX   = 256,   // >= max(HIDDEN, INTER); matmul counter / NSB
    parameter integer PE_M   = 1,    // token ROWS (batch B) sharing one weight fetch
    parameter integer ACT_HW = 0,    // silu HW lanes (0 = full) -- result-invariant resource knob (glm_act HW_LANES)
    // rank 9 (docs/ULTRA_PERF_MODULES.md): 0 = committed serial gate-then-up
    //   passes; 1 = a SECOND matmul engine runs the up GEMV concurrently with the
    //   gate GEMV, so S_UPP/S_UP/S_UPW are never entered.  Measured ceiling at the
    //   profiled slice: 23 cyc x 160 groups = 3,680 cycles.
    //   Default 0 is netlist-identical to the committed module -- the committed
    //   control block is kept VERBATIM in the GU_CONC==0 generate branch, because
    //   folding a `GU_CONC ?` term into the FSM is the shape that has broken
    //   default netlist identity three times in this repo.
    parameter integer GU_CONC = 0,
    // ---- HDR_LATE : L3 header path -- both engines consume stub-fed fw headers
    parameter integer HDR_LATE = 0
)(
    input  wire                     clk,
    input  wire                     rst,        // sync, active-high

    input  wire                     start,
    output reg                      busy,
    output reg                      done,       // 1-cycle pulse when y_out valid

    input  wire [16*HIDDEN*PE_M-1:0] x_vec,     // PE_M bf16 tokens, row-major packed

    // ---- weight request (shared by B rows) ----
    output wire                     w_req,
    output wire [1:0]               w_sel,      // 0=GATE(+UP), 2=DOWN
    output wire [$clog2((INTER>HIDDEN?INTER:HIDDEN)/TN+1)-1:0] w_grp,
    output wire [$clog2(KMAX+1)-1:0] w_k,
    // ---- weight response: Q4_K 4-bit codes (combinational, same cycle) ----
    input  wire [4*TN-1:0]          w_q,        // Q4_K W_{gate|down} lanes
    input  wire [4*TN-1:0]          w_q_up,     // Q4_K W_up lanes (gate/up pass)
    // ---- weight response: per-(col,super-block) Q4_K scales (latched at mm start) ----
    input  wire [16*TN*((KMAX+255)/256)-1:0] w_d,       // gate/down fp16 d
    input  wire [16*TN*((KMAX+255)/256)-1:0] w_dmin,    // gate/down fp16 dmin
    input  wire [96*TN*((KMAX+255)/256)-1:0] w_scales,  // gate/down 6-bit scales
    input  wire [16*TN*((KMAX+255)/256)-1:0] w_d_up,    // up fp16 d
    input  wire [16*TN*((KMAX+255)/256)-1:0] w_dmin_up, // up fp16 dmin
    input  wire [96*TN*((KMAX+255)/256)-1:0] w_scales_up,// up 6-bit scales

    output reg  [16*HIDDEN*PE_M-1:0] y_out
);
    `include "glm_fp.vh"

    localparam integer KW    = $clog2(KMAX+1);
    localparam integer NSB   = (KMAX + 255) / 256;
    localparam integer NG_GU = (INTER  + TN - 1) / TN;   // gate/up output groups
    localparam integer NG_D  = (HIDDEN + TN - 1) / TN;   // down output groups
    localparam integer GW    = $clog2((INTER>HIDDEN?INTER:HIDDEN)/TN + 1);
    localparam integer MTN   = PE_M * TN;
    localparam DN_FULL = (HIDDEN % TN == 0);
    localparam GU_FULL = (INTER  % TN == 0);
    localparam [1:0] SEL_GATE = 2'd0, SEL_DOWN = 2'd2;

    // ---- token + intermediate buffers (per row) ----
    reg [15:0] xbuf [0:PE_M-1][0:HIDDEN-1];
    reg [15:0] hbuf [0:PE_M-1][0:INTER-1];
    integer xr, xk;
    always @(posedge clk)
        if (start)
            for (xr = 0; xr < PE_M; xr = xr + 1)
                for (xk = 0; xk < HIDDEN; xk = xk + 1)
                    xbuf[xr][xk] <= x_vec[16*(HIDDEN*xr + xk) +: 16];

    // ---- FSM ----
    localparam [3:0] S_IDLE=4'd0, S_GATEP=4'd1, S_GATE=4'd2, S_GATEW=4'd3,
                     S_UPP=4'd4, S_UP=4'd5, S_UPW=4'd6, S_GUW=4'd7,
                     S_DNP=4'd8, S_DN=4'd9, S_DNW=4'd10, S_DONE=4'd11;
    reg [3:0]    state;
    reg [GW-1:0] grp;
    reg [KW-1:0] kcnt;
    reg [1:0]    pass_sel;
    wire [KW-1:0] k_len_gu = HIDDEN[KW-1:0];
    wire [KW-1:0] k_len_dn = INTER [KW-1:0];

    // ---- shared matmul operand + weight-select drive ----
    reg           mm_start;
    reg  [KW-1:0] mm_k_len;
    reg           up_pass;
    wire          g_busy, g_ov;
    wire [16*MTN-1:0] g_c;
    wire [16*PE_M-1:0] a_col;
    wire [4*TN-1:0]  w_q_mm      = up_pass ? w_q_up      : w_q;
    wire [16*TN*NSB-1:0] wd_mm   = up_pass ? w_d_up      : w_d;
    wire [16*TN*NSB-1:0] wdm_mm  = up_pass ? w_dmin_up   : w_dmin;
    wire [96*TN*NSB-1:0] wsc_mm  = up_pass ? w_scales_up : w_scales;
    wire          mm_in_valid;

    /* verilator lint_off UNUSEDSIGNAL */
    glm_matmul_q4k #(.PE_M(PE_M), .PE_N(TN), .KMAX(KMAX), .HDR_LATE(HDR_LATE)) u_mm (
        .clk(clk), .rst(rst), .start(mm_start), .k_len(mm_k_len),
        .w_d(wd_mm), .w_dmin(wdm_mm), .w_scales(wsc_mm),
        .in_valid(mm_in_valid), .a_col(a_col), .w_q(w_q_mm),
        .busy(g_busy), .out_valid(g_ov), .c_out(g_c)
    );
    /* verilator lint_on UNUSEDSIGNAL */

    // ---- rank 9: concurrent UP engine (GU_CONC=1 only) ----------------------
    //   Same clk/rst/start/k_len/in_valid/a_col as u_mm, but fed the *_up weight
    //   buses.  Identical k_len and identical accept stream => gu_ov and g_ov
    //   assert on the SAME cycle by construction; the assert below makes a future
    //   divergence loud instead of silent.
    wire                 gu_ov;
    wire [16*MTN-1:0]    gu_c;
    generate
    if (GU_CONC == 0) begin : g_up_none
        assign gu_ov = 1'b0;
        assign gu_c  = {(16*MTN){1'b0}};
    end else begin : g_up_conc
        /* verilator lint_off UNUSEDSIGNAL */
        wire gu_busy;
        glm_matmul_q4k #(.PE_M(PE_M), .PE_N(TN), .KMAX(KMAX), .HDR_LATE(HDR_LATE)) u_mm_up (
            .clk(clk), .rst(rst), .start(mm_start), .k_len(mm_k_len),
            .w_d(w_d_up), .w_dmin(w_dmin_up), .w_scales(w_scales_up),
            .in_valid(mm_in_valid), .a_col(a_col), .w_q(w_q_up),
            .busy(gu_busy), .out_valid(gu_ov), .c_out(gu_c)
        );
        /* verilator lint_on UNUSEDSIGNAL */
        // Simulation-only: a per-cycle $fatal is not synthesizable (yosys:
        // "Can't resolve task name $fatal").  `ifndef YOSYS, not `ifndef SYNTHESIS
        // -- nothing in this repo ever defines SYNTHESIS, so that guard would not
        // actually exclude this from synthesis.  yosys predefines YOSYS.
`ifndef YOSYS
        always @(posedge clk)
            if (!rst && (state == S_GATEW) && (g_ov !== gu_ov))
                $fatal(1, "swiglu GU_CONC: gate/up engines diverged (g_ov=%b gu_ov=%b) -- the concurrency premise is broken", g_ov, gu_ov);
`endif
    end
    endgenerate

    reg [16*MTN-1:0] gate_hold, up_hold;
    reg [GW-1:0]     grp_hold;

    // silu(gate)
    reg               act_in_valid;
    reg  [16*MTN-1:0] act_x_in;
    wire              act_ov;
    wire [16*MTN-1:0] act_y;
    glm_act #(.MODE(1), .LANES(MTN), .HW_LANES(ACT_HW)) u_silu (
        .clk(clk), .rst(rst),
        .in_valid(act_in_valid), .x_in(act_x_in),
        .out_valid(act_ov), .y_out(act_y)
    );

    // ---- combinational operand + weight-request drive ----
    wire stream_gate = (state == S_GATE);
    wire stream_up   = (state == S_UP);
    wire stream_x    = stream_gate | stream_up;
    wire stream_dn   = (state == S_DN);
    localparam integer XIW = (HIDDEN > 1) ? $clog2(HIDDEN) : 1;
    localparam integer HIW = (INTER  > 1) ? $clog2(INTER)  : 1;
    wire [XIW-1:0] x_idx = kcnt[XIW-1:0];
    wire [HIW-1:0] h_idx = kcnt[HIW-1:0];
    genvar ar;
    generate
    for (ar = 0; ar < PE_M; ar = ar + 1) begin : ACOL
        assign a_col[16*ar +: 16] = stream_x  ? xbuf[ar][x_idx] :
                                    stream_dn ? hbuf[ar][h_idx] : 16'b0;
    end
    endgenerate
    assign mm_in_valid = stream_gate | stream_up | stream_dn;
    assign w_req = stream_gate | stream_up | stream_dn;
    assign w_sel = pass_sel;
    assign w_grp = grp[GW-1:0];
    assign w_k   = kcnt;

    generate
    if (GU_CONC == 0) begin : g_fsm_serial
        // DEFAULT: the committed control block, VERBATIM.
        // ---- main control ----
        integer yi, yr;
        always @(posedge clk) begin
            if (rst) begin
                state <= S_IDLE; busy <= 0; done <= 0; grp <= 0; kcnt <= 0;
                pass_sel <= SEL_GATE; up_pass <= 0; mm_start <= 0; mm_k_len <= 0;
                act_in_valid <= 0; act_x_in <= 0; up_hold <= 0; gate_hold <= 0; grp_hold <= 0;
            end else begin
                done <= 0; mm_start <= 0; act_in_valid <= 0;
                case (state)
                S_IDLE: if (start) begin
                    busy <= 1; grp <= 0; state <= S_GATEP; kcnt <= 0;
                    pass_sel <= SEL_GATE; up_pass <= 0; mm_start <= 1; mm_k_len <= k_len_gu;
                end
                S_GATEP: begin kcnt <= 0; state <= S_GATE; end
                S_GATE:  begin if (kcnt == k_len_gu - 1'b1) state <= S_GATEW; kcnt <= kcnt + 1'b1; end
                S_GATEW: if (g_ov) begin
                    gate_hold <= g_c; grp_hold <= grp; state <= S_UPP; kcnt <= 0;
                    up_pass <= 1; mm_start <= 1; mm_k_len <= k_len_gu;
                end
                S_UPP: begin kcnt <= 0; state <= S_UP; end
                S_UP:  begin if (kcnt == k_len_gu - 1'b1) state <= S_UPW; kcnt <= kcnt + 1'b1; end
                S_UPW: if (g_ov) begin
                    act_in_valid <= 1; act_x_in <= gate_hold; up_hold <= g_c;
                    up_pass <= 0; state <= S_GUW;
                end
                S_GUW: if (act_ov) begin
                    if (grp == NG_GU[GW-1:0] - 1'b1) begin
                        state <= S_DNP; grp <= 0; kcnt <= 0; pass_sel <= SEL_DOWN;
                        up_pass <= 0; mm_start <= 1; mm_k_len <= k_len_dn;
                    end else begin
                        grp <= grp + 1'b1; kcnt <= 0; state <= S_GATEP;
                        pass_sel <= SEL_GATE; up_pass <= 0; mm_start <= 1; mm_k_len <= k_len_gu;
                    end
                end
                S_DNP: begin kcnt <= 0; state <= S_DN; end
                S_DN:  begin if (kcnt == k_len_dn - 1'b1) state <= S_DNW; kcnt <= kcnt + 1'b1; end
                S_DNW: if (g_ov) begin
                    for (yr = 0; yr < PE_M; yr = yr + 1)
                        for (yi = 0; yi < TN; yi = yi + 1)
                            if (DN_FULL || (grp*TN + yi < HIDDEN))
                                y_out[16*(HIDDEN*yr + grp*TN + yi) +: 16] <= g_c[16*(yr*TN + yi) +: 16];
                    if (grp == NG_D[GW-1:0] - 1'b1) state <= S_DONE;
                    else begin
                        grp <= grp + 1'b1; kcnt <= 0; state <= S_DNP;
                        pass_sel <= SEL_DOWN; mm_start <= 1; mm_k_len <= k_len_dn;
                    end
                end
                S_DONE: begin done <= 1; busy <= 0; state <= S_IDLE; end
                default: state <= S_IDLE;
                endcase
            end
        end

    end else begin : g_fsm_conc
        // GU_CONC=1: identical except S_GATEW captures both engines.
        // S_UPP/S_UP/S_UPW remain in the encoding (state width unchanged)
        // but are unreachable.
        // ---- main control ----
        integer yi, yr;
        always @(posedge clk) begin
            if (rst) begin
                state <= S_IDLE; busy <= 0; done <= 0; grp <= 0; kcnt <= 0;
                pass_sel <= SEL_GATE; up_pass <= 0; mm_start <= 0; mm_k_len <= 0;
                act_in_valid <= 0; act_x_in <= 0; up_hold <= 0; gate_hold <= 0; grp_hold <= 0;
            end else begin
                done <= 0; mm_start <= 0; act_in_valid <= 0;
                case (state)
                S_IDLE: if (start) begin
                    busy <= 1; grp <= 0; state <= S_GATEP; kcnt <= 0;
                    pass_sel <= SEL_GATE; up_pass <= 0; mm_start <= 1; mm_k_len <= k_len_gu;
                end
                S_GATEP: begin kcnt <= 0; state <= S_GATE; end
                S_GATE:  begin if (kcnt == k_len_gu - 1'b1) state <= S_GATEW; kcnt <= kcnt + 1'b1; end
                // GU_CONC: the up engine finished on this same cycle, so capture BOTH
                // results here and go straight to S_GUW.  up_pass is never raised, so
                // the four operand muxes at :97-100 collapse to the plain gate buses --
                // a mux REMOVED, not added.
                S_GATEW: if (g_ov) begin
`ifdef INJ_GU_WRONGUP
                    // INJECTION: capture the GATE result as `up`, giving
                    // silu(gate)*gate instead of silu(gate)*up.  The numpy golden
                    // MUST catch this -- if it does not, the golden is not
                    // constraining the concurrent engine's VALUE at all and the
                    // whole rank-9 bit-exactness proof is vacuous.
                    gate_hold <= g_c; up_hold <= g_c;  grp_hold <= grp;
`else
                    gate_hold <= g_c; up_hold <= gu_c; grp_hold <= grp;
`endif
                    act_in_valid <= 1; act_x_in <= g_c;
                    state <= S_GUW; kcnt <= 0;
                end
                S_UPP: begin kcnt <= 0; state <= S_UP; end
                S_UP:  begin if (kcnt == k_len_gu - 1'b1) state <= S_UPW; kcnt <= kcnt + 1'b1; end
                S_UPW: if (g_ov) begin
                    act_in_valid <= 1; act_x_in <= gate_hold; up_hold <= g_c;
                    up_pass <= 0; state <= S_GUW;
                end
                S_GUW: if (act_ov) begin
                    if (grp == NG_GU[GW-1:0] - 1'b1) begin
                        state <= S_DNP; grp <= 0; kcnt <= 0; pass_sel <= SEL_DOWN;
                        up_pass <= 0; mm_start <= 1; mm_k_len <= k_len_dn;
                    end else begin
                        grp <= grp + 1'b1; kcnt <= 0; state <= S_GATEP;
                        pass_sel <= SEL_GATE; up_pass <= 0; mm_start <= 1; mm_k_len <= k_len_gu;
                    end
                end
                S_DNP: begin kcnt <= 0; state <= S_DN; end
                S_DN:  begin if (kcnt == k_len_dn - 1'b1) state <= S_DNW; kcnt <= kcnt + 1'b1; end
                S_DNW: if (g_ov) begin
                    for (yr = 0; yr < PE_M; yr = yr + 1)
                        for (yi = 0; yi < TN; yi = yi + 1)
                            if (DN_FULL || (grp*TN + yi < HIDDEN))
                                y_out[16*(HIDDEN*yr + grp*TN + yi) +: 16] <= g_c[16*(yr*TN + yi) +: 16];
                    if (grp == NG_D[GW-1:0] - 1'b1) state <= S_DONE;
                    else begin
                        grp <= grp + 1'b1; kcnt <= 0; state <= S_DNP;
                        pass_sel <= SEL_DOWN; mm_start <= 1; mm_k_len <= k_len_dn;
                    end
                end
                S_DONE: begin done <= 1; busy <= 0; state <= S_IDLE; end
                default: state <= S_IDLE;
                endcase
            end
        end

    end
    endgenerate

    // ---- silu*up merge -> h buffer (bf16(silu(gate) * up)) ----
    reg [15:0] n_hval [0:MTN-1];
    integer mr, mc, mr2, mt;
    always @* begin
        for (mr = 0; mr < PE_M; mr = mr + 1)
            for (mc = 0; mc < TN; mc = mc + 1)
                n_hval[mr*TN + mc] = bf16_mul(act_y[16*(mr*TN+mc) +: 16],
                                              up_hold[16*(mr*TN+mc) +: 16]);
    end
    always @(posedge clk)
        if (act_ov)
            for (mr2 = 0; mr2 < PE_M; mr2 = mr2 + 1)
                for (mt = 0; mt < TN; mt = mt + 1)
                    if (GU_FULL || (grp_hold*TN + mt < INTER))
                        hbuf[mr2][grp_hold*TN + mt] <= n_hval[mr2*TN + mt];
endmodule
/* verilator lint_on DECLFILENAME */
