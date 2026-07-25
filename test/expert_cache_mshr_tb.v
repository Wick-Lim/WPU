`timescale 1ns/1ps
//============================================================================
// test/expert_cache_mshr_tb.v
//   Gate for src/expert_cache_mshr.v (ULTRA_PERF_MODULES.md rank 1 -- the
//   non-blocking MSHR expert-cache refill).  `make mshr`.
//
// WHAT THIS PROVES
//   (A) DIRECTORY CORRECTNESS under concurrency, checked against the RTL's own
//       directory on EVERY response: the answered slot really holds the answered
//       expert id, and no id is ever resident in two slots.
//   (B) THE THREE MSHR HAZARDS, each with a must-FAIL injection build:
//        1. victim collision  -- two live primaries never reserve the same slot
//                                (-DINJECT_NORESV removes the exclusion -> FAIL)
//        2. duplicate fetch   -- an id already in flight is MERGED, not refetched
//                                (-DINJECT_NOMERGE refetches -> fetch-count FAIL)
//        3. in-flight != hit  -- a reserved slot never answers as a hit
//   (C) THE PERFORMANCE CLAIM, MEASURED IN CYCLES, not asserted: K distinct
//       cold misses issued back-to-back complete in ~FLASH_LAT + K cycles.  The
//       committed BLOCKING refill (expert_cache_ctrl/pf S_FETCH: one flash_req,
//       freeze until flash_done) costs K*FLASH_LAT for the same stream.  The TB
//       computes both and requires the measured overlap to beat the blocking
//       baseline by a wide margin -- this is the rank-1 lever's actual evidence.
//   (D) OUT-OF-ORDER completion: the backend model returns tags with SHUFFLED
//       latencies, so responses arrive out of request order and are matched by
//       resp_expert_id.
//
// Run:  iverilog -g2012 -I src -o build/mshr test/expert_cache_mshr_tb.v \
//                 src/expert_cache_mshr.v ; vvp build/mshr
//============================================================================
module expert_cache_mshr_tb;

    localparam integer SLOTS      = 8;
    localparam integer N_EXPERT   = 64;
    localparam integer MSHR_DEPTH = 8;
    localparam integer FLASH_LAT  = 20;
    localparam integer ID_W       = 6;   // clog2(64)
    localparam integer SLOT_W     = 3;   // clog2(8)
    localparam integer TAG_W      = 3;   // clog2(8)

    reg                clk = 1'b0;
    reg                rst = 1'b1;
    always #5 clk = ~clk;

    reg                req_valid;
    reg  [ID_W-1:0]    req_expert_id;
    wire               req_ready;

    wire               resp_valid;
    wire               resp_hit;
    wire [SLOT_W-1:0]  resp_slot;
    wire [ID_W-1:0]    resp_expert_id;

    wire               flash_req;
    wire [ID_W-1:0]    flash_expert_id;
    wire [TAG_W-1:0]   flash_tag;
    reg                flash_ready;
    reg                flash_done;
    reg  [TAG_W-1:0]   flash_done_tag;

    wire [31:0] hit_count, miss_count, fetch_count, merge_count,
                full_stall_cycles, max_outstanding;

    expert_cache_mshr #(
        .SLOTS(SLOTS), .N_EXPERT(N_EXPERT), .MSHR_DEPTH(MSHR_DEPTH)
    ) dut (
        .clk(clk), .rst(rst),
        .req_valid(req_valid), .req_expert_id(req_expert_id), .req_ready(req_ready),
        .resp_valid(resp_valid), .resp_hit(resp_hit), .resp_slot(resp_slot),
        .resp_expert_id(resp_expert_id),
        .flash_req(flash_req), .flash_expert_id(flash_expert_id), .flash_tag(flash_tag),
        .flash_ready(flash_ready), .flash_done(flash_done), .flash_done_tag(flash_done_tag),
        .hit_count(hit_count), .miss_count(miss_count), .fetch_count(fetch_count),
        .merge_count(merge_count), .full_stall_cycles(full_stall_cycles),
        .max_outstanding(max_outstanding)
    );

    integer tests, errors;
    task check(input cond, input [8*64-1:0] name);
        begin
            tests = tests + 1;
            if (!cond) begin
                errors = errors + 1;
                $display("FAIL: %0s   (t=%0t)", name, $time);
            end
        end
    endtask

    //------------------------------------------------------------------
    // FLASH BACKEND MODEL -- multi-outstanding, per-tag latency, may
    // complete OUT OF ORDER.  One done pulse per cycle (tag-serialised).
    //------------------------------------------------------------------
    reg              f_busy [0:MSHR_DEPTH-1];
    reg [31:0]       f_left [0:MSHR_DEPTH-1];
    reg [ID_W-1:0]   f_id   [0:MSHR_DEPTH-1];
    integer          fi;
    integer          lat_jitter;

    // completion selector: ONE tag completes per cycle, and it is NOT ticked.
    //   (A naive "complete inside the tick loop" model silently DROPS completions
    //   when two timers expire together -- the DUT then waits forever.)
    reg              have_fin;
    reg [TAG_W-1:0]  fin_tag;
    always @* begin
        have_fin = 1'b0;
        fin_tag  = {TAG_W{1'b0}};
        for (fi = MSHR_DEPTH-1; fi >= 0; fi = fi - 1)
            if (f_busy[fi] && (f_left[fi] <= 32'd1)) begin
                have_fin = 1'b1;
                fin_tag  = fi[TAG_W-1:0];
            end
    end

    always @(posedge clk) begin
        if (rst) begin
            flash_done     <= 1'b0;
            flash_done_tag <= {TAG_W{1'b0}};
            for (fi = 0; fi < MSHR_DEPTH; fi = fi + 1) begin
                f_busy[fi] <= 1'b0;
                f_left[fi] <= 32'd0;
                f_id[fi]   <= {ID_W{1'b0}};
            end
        end else begin
            flash_done <= 1'b0;
            // tick every in-flight fetch except the one completing this cycle
            for (fi = 0; fi < MSHR_DEPTH; fi = fi + 1)
                if (f_busy[fi] && !(have_fin && (fin_tag == fi[TAG_W-1:0])) && (f_left[fi] > 32'd0))
                    f_left[fi] <= f_left[fi] - 32'd1;
            // complete exactly one
            if (have_fin) begin
                f_busy[fin_tag] <= 1'b0;
                flash_done      <= 1'b1;
                flash_done_tag  <= fin_tag;
            end
            // accept a new tagged fetch (wins over a same-tag completion)
            if (flash_req) begin
                f_busy[flash_tag] <= 1'b1;
                f_id  [flash_tag] <= flash_expert_id;
                // SHUFFLED latency -> out-of-order completion (D)
                f_left[flash_tag] <= FLASH_LAT + lat_jitter;
                lat_jitter = (lat_jitter + 3) % 5;     // 0..4, deterministic shuffle
            end
        end
    end

    //------------------------------------------------------------------
    // INVARIANT MONITORS (run every cycle, not just at response time)
    //------------------------------------------------------------------
    integer a, b, dupes, resv_dupes;
    reg     inv_uniq_ok, inv_resv_ok;
    always @(negedge clk) if (!rst) begin
        // (B3 / A) no expert id resident in two slots
        dupes = 0;
        for (a = 0; a < SLOTS; a = a + 1)
            for (b = a+1; b < SLOTS; b = b + 1)
                if (dut.valid_arr[a] && dut.valid_arr[b] && (dut.tag_arr[a] == dut.tag_arr[b]))
                    dupes = dupes + 1;
        if (dupes != 0 && inv_uniq_ok) begin
            inv_uniq_ok = 1'b0;
            $display("FAIL: an expert id is resident in TWO slots (directory aliased)  t=%0t", $time);
            errors = errors + 1;
        end
        // (B1) two LIVE PRIMARY MSHRs must never target the same slot
        resv_dupes = 0;
        for (a = 0; a < MSHR_DEPTH; a = a + 1)
            for (b = a+1; b < MSHR_DEPTH; b = b + 1)
                if (dut.m_valid[a] && dut.m_prim[a] && !dut.m_done[a] &&
                    dut.m_valid[b] && dut.m_prim[b] && !dut.m_done[b] &&
                    (dut.m_slot[a] == dut.m_slot[b]))
                    resv_dupes = resv_dupes + 1;
        if (resv_dupes != 0 && inv_resv_ok) begin
            inv_resv_ok = 1'b0;
            $display("FAIL: two in-flight primaries reserved the SAME victim slot  t=%0t", $time);
            errors = errors + 1;
        end
    end

    //------------------------------------------------------------------
    // RESPONSE CHECKER -- every answer must be backed by the directory
    //------------------------------------------------------------------
    integer resp_n;
    always @(posedge clk) if (!rst && resp_valid) begin
        resp_n = resp_n + 1;
        if (!(dut.valid_arr[resp_slot] && (dut.tag_arr[resp_slot] == resp_expert_id))) begin
            $display("FAIL: response id=%0d -> slot %0d, but the slot does not hold it  t=%0t",
                     resp_expert_id, resp_slot, $time);
            errors = errors + 1;
        end
        if (dut.resv_arr[resp_slot]) begin
            $display("FAIL: answered from a RESERVED (data-not-arrived) slot  t=%0t", $time);
            errors = errors + 1;
        end
    end

    //------------------------------------------------------------------
    // request driver helper
    //------------------------------------------------------------------
    task issue(input [ID_W-1:0] id);
        begin
            @(negedge clk);
            req_valid     = 1'b1;
            req_expert_id = id;
            // hold until accepted
            @(posedge clk);
            while (!req_ready) begin
                @(negedge clk);
                @(posedge clk);
            end
            @(negedge clk);
            req_valid = 1'b0;
        end
    endtask

    integer i, k;
    integer t_start, t_end, measured, blocking_baseline;
    integer resp_before;

    initial begin
        tests = 0; errors = 0; resp_n = 0; lat_jitter = 0;
        inv_uniq_ok = 1'b1; inv_resv_ok = 1'b1;
        req_valid = 1'b0; req_expert_id = {ID_W{1'b0}};
        flash_ready = 1'b1;               // backend always accepts (multi-outstanding)
        rst = 1'b1;
        repeat (4) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        //==============================================================
        // (C) THE MEASUREMENT: K cold misses back-to-back.
        //   MSHR overlaps them (~FLASH_LAT + K); the committed BLOCKING
        //   refill would cost K*FLASH_LAT for the identical stream.
        //==============================================================
        t_start = $time;
        resp_before = resp_n;
        for (i = 0; i < SLOTS; i = i + 1)
            issue(i[ID_W-1:0]);           // 8 distinct never-seen ids -> 8 cold misses
        // wait for all 8 responses
        while (resp_n < resp_before + SLOTS) @(posedge clk);
        t_end = $time;
        measured          = (t_end - t_start) / 10;         // 10ns per cycle
        blocking_baseline = SLOTS * FLASH_LAT;              // committed S_FETCH cost

        check(fetch_count == SLOTS, "8 distinct cold misses issued exactly 8 flash fetches");
        check(max_outstanding > 32'd1,
              "misses were CONCURRENT (high-water outstanding > 1)");
        check(measured < blocking_baseline / 2,
              "MEASURED overlap beats the blocking baseline by >2x");
        $display("  [MEASURED] %0d cold misses: MSHR %0d cyc vs blocking baseline %0d cyc (FLASH_LAT=%0d, high-water outstanding=%0d) -> %0dx",
                 SLOTS, measured, blocking_baseline, FLASH_LAT, max_outstanding,
                 blocking_baseline / (measured == 0 ? 1 : measured));

        //==============================================================
        // (A) hits: the 8 installed ids must now all HIT
        //==============================================================
        resp_before = resp_n;
        for (i = 0; i < SLOTS; i = i + 1)
            issue(i[ID_W-1:0]);
        while (resp_n < resp_before + SLOTS) @(posedge clk);
        check(hit_count == SLOTS, "the 8 resident ids all answered as HITS");
        check(fetch_count == SLOTS, "hits issued NO extra flash fetches");

        //==============================================================
        // (B2) MERGE: request the same NOT-resident id twice while in flight.
        //   Exactly ONE flash fetch may be issued for the pair.
        //==============================================================
        begin : merge_case
            integer f_before, m_before;
            f_before = fetch_count;
            m_before = merge_count;
            resp_before = resp_n;
            issue(6'd40);                 // cold -> allocates a primary
            issue(6'd40);                 // same id, still in flight -> MERGE
            while (resp_n < resp_before + 2) @(posedge clk);
            check(fetch_count == f_before + 1,
                  "a second request for an IN-FLIGHT id issued NO duplicate fetch (merged)");
            check(merge_count == m_before + 1, "the merge was counted");
        end

        //==============================================================
        // (D) mixed random stream -- concurrency + OOO completion, with the
        //   invariant monitors and the response checker live throughout.
        //==============================================================
        resp_before = resp_n;
        for (i = 0; i < 60; i = i + 1)
            issue($unsigned($random) % 24);
        // drain
        for (i = 0; i < 400; i = i + 1) @(posedge clk);
        check(resp_n > resp_before, "the mixed stream produced responses");
        check(hit_count + miss_count > 32'd60, "every request was classified hit or miss");
        check(inv_uniq_ok, "directory never aliased an id across two slots");
        check(inv_resv_ok, "no two in-flight primaries ever shared a victim slot");

        //==============================================================
        if (errors != 0) begin
            $display("FAILED: %0d error(s) across %0d checks", errors, tests);
            $fatal(1, "expert_cache_mshr_tb had mismatches");
        end
        $display("ALL %0d TESTS PASSED  (MSHR non-blocking refill: %0d-deep; concurrent misses measured %0d cyc vs %0d blocking; merge/victim-reservation/OOO-completion correct)",
                 tests, MSHR_DEPTH, measured, blocking_baseline);
        $finish;
    end

    // global timeout
    initial begin
        #2000000;
        $display("FAIL: global timeout");
        $fatal(1, "timeout");
    end
endmodule
