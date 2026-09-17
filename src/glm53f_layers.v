//============================================================================
// glm53f_layers.v -- the layer walk. ONE glm53f_decoder_block, run L times, with
// the per-layer arm selection and the layer index that annotates every pull.
//
// This is the shape the repo's GLM-5.2 top already uses (glm_model_q4k runs one
// glm_decoder_block_q4k L times and publishes db_layer / db_mode); what is new
// here is that BOTH sites switch, because GLM-5.3-Flash's 45 layers are a mixture
// rather than a prefix.
//
// ---- THE SCHEDULE IS CITED, NOT ASSUMED ----
//   attention:  block l is MLA+DSA iff (l % ATTN_PERIOD) == ATTN_OFFSET
//               [gguf] attention.head_count_kv is a per-block list and [scan]
//               confirms it is strictly periodic: { 3, 7, 11, ... , 43 }, so
//               PERIOD = 4 and OFFSET = 3 -> 11 MLA and 34 KDA out of 45.
//   FFN:        block l is MoE iff l >= N_DENSE
//               [gguf] leading_dense_block_count = 3.
// Both arrive as parameters from configs/full_glm53_flash.vh. Writing either
// pattern as a literal here would turn a checkpoint fact into a source-code fact.
//
// ---- THE RESIDUAL IS INSIDE THE BLOCK, SO THERE IS NO x BETWEEN LAYERS ----
// glm53f_hc_block holds the four mHC streams and every sublayer mixes into them.
// So the walk loads the streams ONCE, runs the block L times without reloading,
// and reads them out at the end. There is no per-layer hidden vector to carry --
// which is a consequence of hyper-connections, not a simplification.
//
// ---- STATE AND WEIGHTS ARE ANNOTATED, NOT STORED ----
// Every pull the block makes comes straight out with `layer` beside it, and the
// KDA state and MLA KV ports pass through untouched. That is the placement
// docs/GLM53_FLASH_PORT.md 4.3y argues for: the KDA state is a full
// read-modify-write of 148 MB that never grows, the KV cache is 11.8 GB read by a
// sparse gather, and the DSA index is a full scan every token -- none of it
// belongs on-die, so none of it is stored here.
//============================================================================
`timescale 1ns/1ps
`ifndef GLM53F_LAYERS_V
`define GLM53F_LAYERS_V

module glm53f_layers #(
    parameter integer L           = 8,   // [gguf] block_count
    parameter integer N_DENSE     = 3,   // [gguf] leading_dense_block_count
    parameter integer ATTN_PERIOD = 4,   // [scan] every Nth block is MLA+DSA
    parameter integer ATTN_OFFSET = 3,   // [scan] ...starting here
    parameter integer LAYW        = (L <= 1) ? 1 : $clog2(L)
)(
    input  wire             clk,
    input  wire             rst,
    input  wire             start,
    output reg              busy,
    output reg              done,

    // ---- the one decoder block this walk drives ----
    output reg              db_start,
    input  wire             db_done,
    output reg              db_streams_load,   // pulsed once, before layer 0

    // ---- what every pull downstream must be annotated with ----
    output reg  [LAYW-1:0]  layer,
    output wire             attn_sel,          // 0 = KDA, 1 = MLA+DSA
    output wire             ffn_sel            // 0 = dense SwiGLU, 1 = MoE
);
`ifndef YOSYS
    initial begin
        if (ATTN_PERIOD < 1 || ATTN_OFFSET >= ATTN_PERIOD)
            $fatal(1, "glm53f_layers: ATTN_OFFSET must be inside [0, ATTN_PERIOD)");
        if (N_DENSE > L)
            $fatal(1, "glm53f_layers: N_DENSE exceeds the layer count");
    end
`endif

    // The schedule, as arithmetic on the CURRENT layer -- so a reader can check it
    // against the config line rather than against a table.
`ifdef INJ_LAYERS_OFF_BY_ONE
    // must FAIL, and this is the QUIET one: shift the phase by a single block, so
    // the MLA blocks become { 2, 6, ..., 42 } instead of { 3, 7, ..., 43 }.
    // Measured at the real L = 45: that is STILL ELEVEN MLA blocks, so every
    // count-based check -- including the census figures N_MLA = 11 / N_KDA = 34 --
    // still agrees, and only the per-layer sequence disagrees. (Dropping the phase
    // entirely, to { 0, 4, ..., 44 }, gives TWELVE and a count check would catch
    // it; that is the loud failure, not this one.)
    assign attn_sel = ((layer % ATTN_PERIOD) == (ATTN_OFFSET[LAYW-1:0] - 1'b1));
`else
    assign attn_sel = ((layer % ATTN_PERIOD) == ATTN_OFFSET[LAYW-1:0]);
`endif
`ifdef INJ_LAYERS_DENSE_OFF
    // must FAIL: every block MoE, i.e. the dense front forgotten.
    assign ffn_sel = 1'b1;
`else
    assign ffn_sel = (layer >= N_DENSE[LAYW-1:0]);
`endif

    localparam [1:0] S_IDLE = 2'd0, S_RUN = 2'd1, S_WAIT = 2'd2, S_FIN = 2'd3;
    reg [1:0] st;

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0;
            db_start <= 1'b0; db_streams_load <= 1'b0; layer <= {LAYW{1'b0}};
        end else begin
            done <= 1'b0; db_start <= 1'b0; db_streams_load <= 1'b0;
            case (st)
                S_IDLE: if (start) begin
                    busy <= 1'b1;
                    layer <= {LAYW{1'b0}};
                    // the mHC streams are loaded ONCE; the block carries them
                    db_streams_load <= 1'b1;
                    st <= S_RUN;
                end
                // one cycle after the load, with layer/attn_sel/ffn_sel settled
                S_RUN: begin db_start <= 1'b1; st <= S_WAIT; end
                S_WAIT: if (db_done) begin
                    if (layer == L[LAYW-1:0] - 1'b1) st <= S_FIN;
                    else begin layer <= layer + 1'b1; st <= S_RUN; end
                end
                S_FIN: begin done <= 1'b1; busy <= 1'b0; st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
`endif // GLM53F_LAYERS_V
