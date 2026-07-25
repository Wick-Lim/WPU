`timescale 1ns/1ps
//============================================================================
// test/ddr5_xbar_lanes_tb.v
//   Gate for docs/ULTRA_PERF_MODULES.md rank 2 -- "turn N_CH into SUSTAINED
//   N beats/cycle, not just latency-hiding".  `make rank2`.
//
// THE CLAIM UNDER TEST
//   The committed fabric has N_CH channel pipelines but ONE response grant per
//   cycle.  So N_CH buys outstanding depth (a channel can be busy) while the
//   sustained delivery stays 1 beat/cycle -- one channel's worth -- and the
//   other N_CH-1 completed reads simply back up in their FIFOs.  That response
//   arbiter, not the channel count, is the throttle.  RESP_LANES>1 drains up to
//   that many DISTINCT channels per cycle.
//
// METHOD  (measure, don't assert)
//   Every channel is kept SUPPLIED: the stub returns a completion on every
//   channel every cycle it is accepted, so the FIFOs stay non-empty and the
//   drain rate is limited by the arbiter alone -- which is exactly the quantity
//   in question.  We then count beats delivered per cycle over a long window.
//     RESP_LANES=1 must measure ~1.0 beats/cycle  (the committed ceiling)
//     RESP_LANES=L must measure ~L                (the lever actually working)
//   Data integrity is checked on every drained beat: each beat carries its
//   channel id and a per-channel sequence number, so a beat delivered twice, a
//   beat lost, or two lanes drained from the SAME channel in one cycle is
//   caught -- that last one is the specific hazard multi-drain introduces.
//============================================================================
module ddr5_xbar_lanes_tb;

    localparam integer N_CH   = 8;
    localparam integer DATA_W = 32;
    localparam integer TAG_W  = 8;
    localparam integer ADDR_W = 16;
    localparam integer RESP_QD = 4;
`ifdef TB_RESP_LANES
    localparam integer LANES = `TB_RESP_LANES;
`else
    localparam integer LANES = 1;
`endif
    localparam integer WINDOW = 500;      // measurement window in cycles

    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst;

    // requester side is idle -- this gate measures the RESPONSE drain only
    wire                     req_ready;
    wire [N_CH-1:0]          mem_req_valid;
    wire [N_CH*ADDR_W-1:0]   mem_req_addr;
    wire [N_CH*TAG_W-1:0]    mem_req_tag;

    reg  [N_CH-1:0]          mem_resp_valid;
    wire [N_CH-1:0]          mem_resp_ready;
    reg  [N_CH*DATA_W-1:0]   mem_resp_data;
    reg  [N_CH*TAG_W-1:0]    mem_resp_tag;

    wire [LANES-1:0]         resp_valid;
    wire [LANES*DATA_W-1:0]  resp_data;
    wire [LANES*TAG_W-1:0]   resp_tag;

    ddr5_xbar #(
        .N_CH(N_CH), .RESP_LANES(LANES), .ADDR_W(ADDR_W), .DATA_W(DATA_W),
        .TAG_W(TAG_W), .RESP_QD(RESP_QD), .BANK_LSB(0)
    ) dut (
        .clk(clk), .rst(rst),
        .req_valid(1'b0), .req_ready(req_ready), .req_addr({ADDR_W{1'b0}}), .req_tag({TAG_W{1'b0}}),
        .mem_req_valid(mem_req_valid), .mem_req_ready({N_CH{1'b1}}),
        .mem_req_addr(mem_req_addr), .mem_req_tag(mem_req_tag),
        .mem_resp_valid(mem_resp_valid), .mem_resp_ready(mem_resp_ready),
        .mem_resp_data(mem_resp_data), .mem_resp_tag(mem_resp_tag),
        .resp_valid(resp_valid), .resp_ready(1'b1),
        .resp_data(resp_data), .resp_tag(resp_tag)
    );

    // ---- per-channel supply: a fresh completion whenever the FIFO takes one --
    //   beat payload = {channel, per-channel sequence} so every beat is unique.
    integer c;
    reg [15:0] seq_tx [0:N_CH-1];      // next sequence to hand to channel c
    reg [15:0] seq_rx [0:N_CH-1];      // next sequence expected out of channel c
    integer    beats_out;
    integer    cycles;
    integer    errors, tests;

    always @(posedge clk) begin
        if (rst) begin
            mem_resp_valid <= {N_CH{1'b0}};
            for (c = 0; c < N_CH; c = c + 1) begin
                seq_tx[c] <= 16'd0;
                mem_resp_data[c*DATA_W +: DATA_W] <= {DATA_W{1'b0}};
                mem_resp_tag [c*TAG_W  +: TAG_W ] <= {TAG_W{1'b0}};
            end
        end else begin
            for (c = 0; c < N_CH; c = c + 1) begin
                // keep every channel supplied so the drain is the only limiter
                mem_resp_valid[c] <= 1'b1;
                if (mem_resp_valid[c] && mem_resp_ready[c]) begin
                    seq_tx[c] <= seq_tx[c] + 16'd1;
                    mem_resp_data[c*DATA_W +: DATA_W] <= {c[15:0], seq_tx[c] + 16'd1};
                    mem_resp_tag [c*TAG_W  +: TAG_W ] <= c[TAG_W-1:0];
                end else if (seq_tx[c] == 16'd0) begin
                    mem_resp_data[c*DATA_W +: DATA_W] <= {c[15:0], 16'd0};
                    mem_resp_tag [c*TAG_W  +: TAG_W ] <= c[TAG_W-1:0];
                end
            end
        end
    end

    // ---- drain checker: order per channel, no duplicate, no same-channel pair -
    integer li, lj, ch, sq;
    reg [N_CH-1:0] seen_this_cycle;
    always @(posedge clk) if (!rst) begin
        cycles = cycles + 1;
        seen_this_cycle = {N_CH{1'b0}};
        for (li = 0; li < LANES; li = li + 1) begin
            if (resp_valid[li]) begin
                beats_out = beats_out + 1;
                ch = resp_data[li*DATA_W + 16 +: 16];
                sq = resp_data[li*DATA_W      +: 16];
                if (ch >= N_CH) begin
                    if (errors < 5) $display("FAIL: lane %0d delivered impossible channel %0d", li, ch);
                    errors = errors + 1;
                end else begin
                    // the SAME channel must never be drained twice in one cycle
                    if (seen_this_cycle[ch]) begin
                        if (errors < 5)
                            $display("FAIL: channel %0d drained by TWO lanes in one cycle (multi-drain hazard)", ch);
                        errors = errors + 1;
                    end
                    seen_this_cycle[ch] = 1'b1;
                    // and each channel's beats must come out in order, none skipped
                    if (sq !== seq_rx[ch]) begin
                        if (errors < 5)
                            $display("FAIL: channel %0d out of order: got seq %0d, expected %0d", ch, sq, seq_rx[ch]);
                        errors = errors + 1;
                    end
                    seq_rx[ch] = seq_rx[ch] + 1;
                end
            end
        end
    end

    real rate;
    initial begin
        errors = 0; tests = 0; beats_out = 0; cycles = 0;
        for (c = 0; c < N_CH; c = c + 1) seq_rx[c] = 16'd0;
        rst = 1'b1;
        repeat (4) @(posedge clk);
        rst = 1'b0;
        // let the pipeline fill, then measure a clean window
        repeat (30) @(posedge clk);
        beats_out = 0; cycles = 0;
        repeat (WINDOW) @(posedge clk);

        rate = (cycles > 0) ? (1.0 * beats_out) / (1.0 * cycles) : 0.0;
        $display("  [MEASURED] N_CH=%0d RESP_LANES=%0d: %0d beats in %0d cycles -> %.3f beats/cycle",
                 N_CH, LANES, beats_out, cycles, rate);

        tests = tests + 1;
        if (errors != 0) $display("FAIL: %0d integrity error(s) during the drain", errors);

        tests = tests + 1;
        if (rate < (0.90 * LANES)) begin
            $display("FAIL: sustained drain %.3f beats/cycle is below the %0d-lane expectation", rate, LANES);
            errors = errors + 1;
        end

        if (errors != 0) begin
            $display("FAILED: %0d error(s) across %0d checks", errors, tests);
            $fatal(1, "ddr5_xbar_lanes_tb had mismatches");
        end
        $display("ALL %0d TESTS PASSED  (fabric drain: %.3f beats/cycle at RESP_LANES=%0d, per-channel order kept, no channel double-drained)",
                 tests, rate, LANES);
        $finish;
    end

    initial begin
        #200000;
        $display("FAIL: global timeout");
        $fatal(1, "timeout");
    end
endmodule
