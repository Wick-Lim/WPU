`timescale 1ns/1ps
//============================================================================
// test/cdc_wide_beat_tb.v
//   Gate for docs/ULTRA_PERF_MODULES.md rank 12 -- "cross the clock domain at
//   the WIDE beat, and size the FIFO so it never back-pressures".  `make rank12`.
//
// WHY THIS EXISTS
//   Ranks 2-3 widen the read fabric to N lanes.  If the core<->die CDC crossing
//   between that fabric and the die stays a narrow, shallow FIFO, it re-chokes
//   the stream to one word per cycle and the whole widening is wasted.  Two
//   separate properties have to hold, and only one of them is obvious:
//     (a) WIDTH  -- a wide beat must cross ATOMICALLY: all N lanes of beat i
//         arrive together, in order, uncorrupted.  cdc_async_fifo is already
//         DATA_W-parameterized, so this is a property to CONFIRM, not to build.
//     (b) DEPTH  -- an async gray-pointer FIFO only sustains one beat per cycle
//         if its depth covers the cross-domain round trip (two 2-FF
//         synchronizers plus the flag registration).  Too shallow and `full` /
//         `empty` oscillate: the crossing silently delivers LESS than one beat
//         per cycle even though nothing is dropped and no test fails.
//   This TB measures (b) -- it reports sustained beats/cycle per depth and
//   asserts the fabric-width crossing reaches ~1.0 -- and checks (a) exactly.
//
// METHOD
//   DATA_W = LANES*LANE_W (8 x 32b = 256b, a fabric-shaped beat).  Each beat is
//   lane-tagged: lane L of beat i carries {beat_id, lane_id} so a lane swap, a
//   lane stuck at its previous value, or a torn beat is caught per lane rather
//   than merely as "some data differs".
//   The READ clock is deliberately FASTER than the write clock, so the reader is
//   never the bottleneck and the measured write-side throughput isolates the
//   CDC's OWN limit.  The writer asserts wr_en whenever !full; sustained
//   throughput = beats accepted / write cycles elapsed.
//
// INJECTION (proves the throughput assert is load-bearing)
//   -DINJECT_SHALLOW builds the SAME test with a depth-4 FIFO.  Four entries
//   cannot cover the two 2-FF synchronizer hops plus flag registration, so `full`
//   throttles the writer and sustained throughput must fall under the threshold:
//   the gate MUST FAIL.  If it passes, the measurement constrains nothing.
//   (Depth 2 -- ADDR_W=1 -- is not even elaboratable: cdc_async_fifo indexes
//   rgray_wq2[ADDR_W-1:0], so ADDR_W>=2 is a structural floor of the module.)
//============================================================================
module cdc_wide_beat_tb;

    localparam integer LANES  = 8;
    localparam integer LANE_W = 32;
    localparam integer DATA_W = LANES * LANE_W;      // 256b fabric beat
`ifdef INJECT_SHALLOW
    localparam integer ADDR_W = 2;                   // depth 4 -- too shallow for the round trip
`else
    localparam integer ADDR_W = 4;                   // depth 16
`endif
    localparam integer NBEATS = 400;

    // read clock FASTER than write clock -> the reader is never the bottleneck,
    // so what we measure is the crossing's own sustained rate.
    reg wclk = 1'b0;  always #5.0 wclk = ~wclk;      // 10.0 ns
    reg rclk = 1'b0;  always #4.3 rclk = ~rclk;      // 8.6 ns, asynchronous

    reg                wrst_n, rrst_n;
    reg                wr_en;
    reg  [DATA_W-1:0]  wr_data;
    wire               full;
    reg                rd_en;
    wire [DATA_W-1:0]  rd_data;
    wire               empty;

    cdc_async_fifo #(.DATA_W(DATA_W), .ADDR_W(ADDR_W)) dut (
        .wclk(wclk), .wrst_n(wrst_n), .wr_en(wr_en), .wr_data(wr_data), .full(full),
        .rclk(rclk), .rrst_n(rrst_n), .rd_en(rd_en), .rd_data(rd_data), .empty(empty)
    );

    // lane-tagged beat: lane L of beat i = {i[15:0], L[15:0]}
    function [DATA_W-1:0] mk_beat(input integer i);
        integer L;
        begin
            mk_beat = {DATA_W{1'b0}};
            for (L = 0; L < LANES; L = L + 1)
                mk_beat[L*LANE_W +: LANE_W] = {i[15:0], L[15:0]};
        end
    endfunction

    integer tests, errors;
    integer w_i, r_i;              // beats written / read
    integer w_cycles;              // write-clock cycles from first accept to last
    integer started;

    task check(input cond, input [8*72-1:0] name);
        begin
            tests = tests + 1;
            if (!cond) begin errors = errors + 1; $display("FAIL: %0s", name); end
        end
    endtask

    // ---------------- writer: push whenever not full ----------------
    //   Drive wr_en/wr_data in the NEGEDGE half from the CURRENT `full` (the same
    //   discipline the verified cdc_async_fifo_tb uses).  Registering them a cycle
    //   early would let `full` change underneath and silently drop a beat while the
    //   TB counted it as written -- which is a TB bug, not a DUT bug.
    initial begin
        wr_en = 1'b0; wr_data = {DATA_W{1'b0}};
        forever begin
            @(negedge wclk);
            if (wrst_n && !full && (w_i < NBEATS)) begin
                wr_en   = 1'b1;
                wr_data = mk_beat(w_i);
            end else begin
                wr_en   = 1'b0;
            end
            @(posedge wclk);
            if (wr_en && !full) begin
                w_i = w_i + 1;
                if (!started) started = 1;
            end
            if (started && (w_i < NBEATS)) w_cycles = w_cycles + 1;
        end
    end

    // ---------------- reader: pop whenever not empty, check every lane -------
    //   Protocol (from the verified cdc_async_fifo_tb): drive rd_en in the NEGEDGE
    //   half from `empty`; on the posedge where (rd_en & ~empty) holds the head
    //   word appears on rd_data just after that edge -- sample it #1 later.
    integer L;
    reg [DATA_W-1:0] exp;
    initial begin
        rd_en = 1'b0;
        forever begin
            @(negedge rclk);
            rd_en = rrst_n && !empty;
            @(posedge rclk);
            if (rd_en && !empty) begin
                #1;
                exp = mk_beat(r_i);
                for (L = 0; L < LANES; L = L + 1) begin
                    if (rd_data[L*LANE_W +: LANE_W] !== exp[L*LANE_W +: LANE_W]) begin
                        if (errors < 6)
                            $display("FAIL: beat %0d lane %0d = %h, expected %h (torn/swapped/stale lane)",
                                     r_i, L, rd_data[L*LANE_W +: LANE_W], exp[L*LANE_W +: LANE_W]);
                        errors = errors + 1;
                    end
                end
                r_i = r_i + 1;
            end
        end
    end

    real thru;
    initial begin
        tests = 0; errors = 0; w_i = 0; r_i = 0; w_cycles = 0; started = 0;
        wr_en = 0; wr_data = 0;
        wrst_n = 1'b0; rrst_n = 1'b0;
        repeat (6) @(posedge wclk);
        wrst_n = 1'b1; rrst_n = 1'b1;

        // run until every beat has crossed (or timeout)
        while (r_i < NBEATS) @(posedge rclk);
        repeat (20) @(posedge rclk);

        // ---- (a) WIDTH: every lane of every beat arrived exactly, in order ----
        check(errors == 0, "all 8 lanes of all beats crossed atomically, in order");
        check(r_i == NBEATS, "no beat lost or duplicated across the domain");

        // ---- (b) DEPTH: sustained beats per write cycle ----
        thru = (w_cycles > 0) ? (1.0 * NBEATS) / (1.0 * w_cycles) : 0.0;
        $display("  [MEASURED] DATA_W=%0d (%0d lanes x %0db), depth=%0d: %0d beats in %0d write cycles -> %.3f beats/cycle",
                 DATA_W, LANES, LANE_W, (1<<ADDR_W), NBEATS, w_cycles, thru);
        check(thru > 0.95,
              "the crossing SUSTAINS ~1 wide beat per cycle (depth covers the sync round trip)");

        if (errors != 0) begin
            $display("FAILED: %0d error(s) across %0d checks", errors, tests);
            $fatal(1, "cdc_wide_beat_tb had mismatches");
        end
        $display("ALL %0d TESTS PASSED  (wide-beat CDC: %0d-lane atomic crossing, %.3f beats/cycle sustained at depth %0d)",
                 tests, LANES, thru, (1<<ADDR_W));
        $finish;
    end

    initial begin
        #900000;
        $display("FAIL: global timeout (the crossing stalled)");
        $fatal(1, "timeout");
    end
endmodule
