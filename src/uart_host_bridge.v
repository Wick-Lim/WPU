// ============================================================================
// uart_host_bridge.v -- UART <-> glm_q4k_system_cdc host protocol  (L3 S6)
//
//   The CDC top's host face is small: start(pulse) + prompt_tok/start_pos/s_len
//   in, busy/done/next_tok/tok_valid out.  A board needs SOMETHING to drive it;
//   the cheapest thing every XCKU3P-class board and every host PC has is a UART.
//   PCIe/DMA is a later upgrade, not a bring-up need -- at L3 rates (single-digit
//   tok/s on a reduced config) even 115200 baud is three orders of magnitude
//   more than the token stream needs.
//
//   WIRE PROTOCOL (fixed frames, 8N1):
//     host -> fpga : 'T'  tok[15:8] tok[7:0]  pos[15:8] pos[7:0]  slen   (6 bytes)
//                    -> pulses start with {prompt_tok, start_pos, s_len}
//     fpga -> host : 'K'  tok[15:8] tok[7:0]                             (3 bytes)
//                    -> sent once per tok_valid pulse
//   Any byte other than 'T' in the idle state is ignored (resync is trivial:
//   the parser only leaves idle on 'T').  Widths beyond 16/8 bits are truncated
//   -- the L3 reduced config is far below either limit, and the widths are
//   parameters if that ever changes.
//
//   CLOCKING.  This module lives ENTIRELY in the host clock domain, exactly where
//   the CDC top's host-side ports already are; the host<->core crossing stays the
//   CDC top's own guarded crossing.  No new clock domains are introduced here.
//
//   NOT VERIFIED HERE (board-only, stated): electrical levels, the actual pin,
//   baud-rate error against a real oscillator.  The gate proves framing, parsing,
//   start-pulse generation and token return over REAL 8N1 waveforms at the
//   parameterised divisor, plus a mid-bit-sampling injection.
// ============================================================================
`default_nettype none

module uart_host_bridge #(
    parameter integer CLK_DIV = 434,   // clk / baud (e.g. 50 MHz / 115200)
    parameter integer TOKW    = 16,
    parameter integer POSW    = 16,
    parameter integer SLENW   = 8
)(
    input  wire             clk,
    input  wire             rst,

    // ---- UART pins ----
    input  wire             uart_rx,
    output reg              uart_tx,

    // ---- host face of glm_q4k_system_cdc ----
    output reg              start,        // 1-cycle pulse
    output reg [TOKW-1:0]   prompt_tok,
    output reg [POSW-1:0]   start_pos,
    output reg [SLENW-1:0]  s_len,
    input  wire             tok_valid,    // 1-cycle pulse
    input  wire [TOKW-1:0]  next_tok
);

    localparam integer DIVW = $clog2(CLK_DIV);
    //   Counter reloads are CLK_DIV-1 (and CLK_DIV/2-1), NOT CLK_DIV: a down-counter
    //   checked at zero consumes reload+1 cycles, so reloading CLK_DIV gives 17-cycle
    //   bits at CLK_DIV=16 -- one cycle of drift per bit, half a bit by bit 8, and the
    //   last data bit lands in the stop bit.  Measured as a hard gate failure, not styled.

    // ------------------------- RX: 8N1 receiver ------------------------------
    //   Majority-free single sample at MID-BIT: start edge arms a half-period
    //   count, then 8 samples one period apart.  Mid-bit is the entire trick --
    //   sampling at the edge reads the PREVIOUS bit half the time (the injection
    //   below proves the gate can tell).
    reg [1:0]      rx_sync;
    always @(posedge clk) rx_sync <= {rx_sync[0], uart_rx};
    wire rxs = rx_sync[1];

    localparam [1:0] R_IDLE=2'd0, R_START=2'd1, R_BITS=2'd2, R_STOP=2'd3;
    reg [1:0]      rst_r;
    reg [DIVW:0]   rcnt;
    reg [2:0]      rbit;
    reg [7:0]      rsh;
    reg            rx_stb;          // 1-cycle: rsh holds a received byte
    always @(posedge clk) begin
        if (rst) begin
            rst_r <= R_IDLE; rcnt <= 0; rbit <= 0; rsh <= 0; rx_stb <= 1'b0;
        end else begin
            rx_stb <= 1'b0;
            case (rst_r)
            R_IDLE:  if (!rxs) begin rst_r <= R_START; rcnt <= CLK_DIV[DIVW:0]/2 - 1'b1; end
            R_START: if (rcnt == 0) begin
                         if (!rxs) begin rst_r <= R_BITS; rcnt <= CLK_DIV[DIVW:0] - 1'b1; rbit <= 0; end
                         else       rst_r <= R_IDLE;      // glitch, not a start bit
                     end else rcnt <= rcnt - 1'b1;
            R_BITS:  if (rcnt == 0) begin
`ifdef INJ_UART_EDGE
                         // INJECTION (never a normal build): re-arm the counter
                         //   with HALF a period, so every sample lands at a bit
                         //   EDGE instead of mid-bit.  Framing must collapse and
                         //   the gate's byte comparisons MUST fail.
                         rcnt <= CLK_DIV[DIVW:0]/2;
`else
                         rcnt <= CLK_DIV[DIVW:0] - 1'b1;
`endif
                         rsh  <= {rxs, rsh[7:1]};          // LSB first
                         rbit <= rbit + 1'b1;
                         if (rbit == 3'd7) rst_r <= R_STOP;
                     end else rcnt <= rcnt - 1'b1;
            R_STOP:  if (rcnt == 0) begin
                         if (rxs) rx_stb <= 1'b1;          // valid stop bit
                         rst_r <= R_IDLE;
                     end else rcnt <= rcnt - 1'b1;
            endcase
        end
    end

    // ------------------------- command parser --------------------------------
    localparam [2:0] P_IDLE=3'd0, P_T1=3'd1, P_T0=3'd2, P_P1=3'd3, P_P0=3'd4, P_SL=3'd5;
    reg [2:0] pst;
    always @(posedge clk) begin
        if (rst) begin
            pst <= P_IDLE; start <= 1'b0;
            prompt_tok <= 0; start_pos <= 0; s_len <= 0;
        end else begin
            start <= 1'b0;
            if (rx_stb) case (pst)
                P_IDLE: if (rsh == "T") pst <= P_T1;
                P_T1:   begin prompt_tok[TOKW-1:8] <= rsh[TOKW-9:0]; pst <= P_T0; end
                P_T0:   begin prompt_tok[7:0]      <= rsh;           pst <= P_P1; end
                P_P1:   begin start_pos[POSW-1:8]  <= rsh[POSW-9:0]; pst <= P_P0; end
                P_P0:   begin start_pos[7:0]       <= rsh;           pst <= P_SL; end
                P_SL:   begin s_len <= rsh[SLENW-1:0]; start <= 1'b1; pst <= P_IDLE; end
                default: pst <= P_IDLE;
            endcase
        end
    end

    // ------------------------- TX: 8N1 transmitter + token framer ------------
    //   A tok_valid pulse queues a 3-byte 'K' frame.  A 4-deep token FIFO absorbs
    //   the (astronomically unlikely at L3 rates) case of tokens arriving faster
    //   than the UART drains.
    reg [TOKW-1:0] tfifo [0:3];
    reg [1:0]      t_wp, t_rp;
    reg [2:0]      t_occ;

    localparam [1:0] T_IDLE=2'd0, T_SHIFT=2'd1;
    reg [1:0]      tst;
    reg [9:0]      tsh;            // {stop, data[7:0], start}
    reg [3:0]      tbit;
    reg [DIVW:0]   tcnt;
    reg [1:0]      tbyte;          // which byte of the frame (0:'K' 1:hi 2:lo)
    reg [TOKW-1:0] tok_q;
    reg            t_push, t_pop;

    always @(posedge clk) begin
        if (rst) begin
            uart_tx <= 1'b1; tst <= T_IDLE; tsh <= 10'h3FF; tbit <= 0; tcnt <= 0;
            t_wp <= 0; t_rp <= 0; t_occ <= 0; tbyte <= 0; tok_q <= 0;
        end else begin
            //   push/pop may coincide (a token arrives on the cycle a frame
            //   finishes).  Two separate `t_occ <=` in one always block would
            //   lose one update -- last assignment wins -- so occupancy is
            //   computed ONCE from explicit push/pop strobes.
            t_push = tok_valid && (t_occ != 3'd4);
            t_pop  = (tst == T_SHIFT) && (tcnt == 0) && (tbit == 4'd1) && (tbyte == 2'd2);
            if (t_push) begin
                tfifo[t_wp] <= next_tok;
                t_wp  <= t_wp + 1'b1;
            end
            t_occ <= t_occ + (t_push ? 3'd1 : 3'd0) - (t_pop ? 3'd1 : 3'd0);
            case (tst)
            T_IDLE: if (t_occ != 0 || tbyte != 0) begin
                        if (tbyte == 0) begin tok_q <= tfifo[t_rp]; end
                        tsh  <= (tbyte == 0) ? {1'b1, "K", 1'b0}
                              : (tbyte == 1) ? {1'b1, tok_q[TOKW-1:8], 1'b0}
                                             : {1'b1, tok_q[7:0], 1'b0};
                        tbit <= 4'd10;
                        tcnt <= CLK_DIV[DIVW:0] - 1'b1;
                        tst  <= T_SHIFT;
                    end
            T_SHIFT: begin
                        uart_tx <= tsh[0];
                        if (tcnt == 0) begin
                            tsh  <= {1'b1, tsh[9:1]};
                            tcnt <= CLK_DIV[DIVW:0] - 1'b1;
                            tbit <= tbit - 1'b1;
                            if (tbit == 4'd1) begin
                                tst <= T_IDLE;
                                if (tbyte == 2'd2) begin
                                    tbyte <= 0;
                                    t_rp  <= t_rp + 1'b1;
                                    // t_occ handled by the single push/pop update
                                end else tbyte <= tbyte + 1'b1;
                            end
                        end else tcnt <= tcnt - 1'b1;
                     end
            endcase
        end
    end

endmodule
`default_nettype wire
