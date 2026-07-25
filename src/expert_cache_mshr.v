`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// expert_cache_mshr.v  --  NON-BLOCKING (MSHR) EXPERT-WEIGHT CACHE REFILL
//                          ULTRA_PERF_MODULES.md rank 1
//----------------------------------------------------------------------------
// WHY THIS EXISTS
//   The committed demand cache (expert_cache_ctrl / expert_cache_pf) services a
//   miss with a BLOCKING refill: S_FETCH issues one flash_req and freezes until
//   flash_done.  A token's top-k routed experts therefore pay their miss
//   latencies END TO END -- k misses cost ~k*FLASH_LAT -- and the demand
//   bandwidth the die can express is capped at 1/FLASH_LAT regardless of how
//   many channels the fabric has.  That single-outstanding refill is the first
//   throttle on the fetch path (docs/ULTRA_PERF_MODULES.md rank 1).
//
//   This module is the non-blocking replacement: an M-entry MSHR (Miss Status
//   Holding Register) file lets M misses be in flight at once, so the k misses
//   OVERLAP and cost ~FLASH_LAT + k instead of k*FLASH_LAT.  The concurrency
//   win is MEASURED in cycles by test/expert_cache_mshr_tb.v (`make mshr`),
//   not asserted here.
//
// SCOPE / HONESTY  (read this before quoting a speedup)
//   * This module is NOT yet wired into the product top.  The committed
//     datapath (glm_q4k_system -> expert_cache_pf) is untouched and its netlist
//     is unchanged -- zero risk to the verified build.
//   * Wiring it in requires widening TWO more single-outstanding seams that sit
//     around it (both found by the same review):
//       (a) the request issuer in glm_q4k_system.v (~:857) gates new requests on
//           `awaiting`, i.e. exactly ONE outstanding request to the cache;
//       (b) the single flash arbiter in glm_q4k_system.v (~:1000) holds `fl_busy`
//           for one untagged transaction at a time, shared with the KV pager.
//     Until those are widened the MSHR depth cannot be exercised in the system.
//     The end-to-end tok/s effect is therefore [EST] until measured on the
//     integrated top; what is MEASURED here is the mechanism's cycle overlap.
//
// PROTOCOL  (a superset of the committed cache's; new signals are marked NEW)
//   REQUEST : req_valid/req_expert_id with req_ready (NEW) -- the module accepts
//     a request whenever it has an MSHR entry and (for a miss) a free victim.
//     Multiple requests may be outstanding; there is no busy/stall handshake.
//   RESPONSE: resp_valid/resp_hit/resp_slot plus resp_expert_id (NEW).  Because
//     several requests are in flight and the backend may complete OUT OF ORDER,
//     the response carries the expert id it belongs to; the consumer matches on
//     it instead of assuming one-at-a-time.  One response per cycle.
//   FLASH   : flash_req/flash_expert_id gain flash_tag (NEW) + flash_ready (NEW,
//     backend accepts this cycle); completion is flash_done + flash_done_tag
//     (NEW).  Tagged completion means the backend may return OUT OF ORDER --
//     the TB exercises both in-order and shuffled returns.
//
// THE THREE CORRECTNESS HAZARDS MSHR INTRODUCES (all handled, all injected-against)
//   1. VICTIM COLLISION -- two concurrent misses must not choose the SAME victim
//      slot.  A slot reserved by an in-flight MSHR is marked resv_arr[] and is
//      excluded from victim selection until its install lands.  (Injection
//      -DINJECT_NORESV removes the exclusion; the TB must FAIL.)
//   2. DUPLICATE FETCH -- a request for an id already in flight must NOT issue a
//      second flash fetch.  It allocates a MERGED entry (no_fetch=1) bound to the
//      primary's slot and completes with it.  (Injection -DINJECT_NOMERGE issues
//      the duplicate; the TB must FAIL on the fetch count.)
//   3. IN-FLIGHT IS NOT A HIT -- a reserved slot's data has not arrived, so a
//      lookup must not report a hit on it.  Tag compare is qualified by
//      valid_arr && !resv_arr.
//
// POLICY
//   EXACT move-to-front LRU, identical in form to the committed cache: rank[s]
//   is slot s's recency position (0 == MRU), {rank} is always a permutation.
//   Touch on a demand hit and on install.  Victim = lowest-index free-and-
//   unreserved invalid slot, else the LRU among unreserved slots.
//
// STYLE: synchronous, ACTIVE-HIGH reset; every reg reset; no latch; no comb loop.
//============================================================================
module expert_cache_mshr #(
    parameter integer SLOTS      = 8,    // cache slots
    parameter integer N_EXPERT   = 64,   // total distinct expert ids
    parameter integer MSHR_DEPTH = 8,    // concurrent misses in flight (>=1)
    // Derived widths (do NOT override) -- guard the degenerate ==1 cases.
    parameter integer ID_W   = (N_EXPERT   <= 1) ? 1 : $clog2(N_EXPERT),
    parameter integer SLOT_W = (SLOTS      <= 1) ? 1 : $clog2(SLOTS),
    parameter integer TAG_W  = (MSHR_DEPTH <= 1) ? 1 : $clog2(MSHR_DEPTH)
)(
    input  wire                clk,
    input  wire                rst,          // synchronous, ACTIVE-HIGH

    // ---- request (non-blocking; multiple may be outstanding) ----
    input  wire                req_valid,
    input  wire [ID_W-1:0]     req_expert_id,
    output wire                req_ready,    // NEW: accepts a request this cycle

    // ---- response (may complete OUT OF ORDER -> carries the id) ----
    output reg                 resp_valid,
    output reg                 resp_hit,
    output reg  [SLOT_W-1:0]   resp_slot,
    output reg  [ID_W-1:0]     resp_expert_id,   // NEW

    // ---- Flash/NVMe DMA, tagged ----
    output reg                 flash_req,
    output reg  [ID_W-1:0]     flash_expert_id,
    output reg  [TAG_W-1:0]    flash_tag,        // NEW
    input  wire                flash_ready,      // NEW: backend accepts this cycle
    input  wire                flash_done,
    input  wire [TAG_W-1:0]    flash_done_tag,   // NEW (OOO completion)

    // ---- stats ----
    output reg  [31:0]         hit_count,
    output reg  [31:0]         miss_count,
    output reg  [31:0]         fetch_count,      // flash fetches ISSUED (merges do not count)
    output reg  [31:0]         merge_count,      // requests that merged onto an in-flight fetch
    output reg  [31:0]         full_stall_cycles,// cycles a request was offered but not acceptable
    output reg  [31:0]         max_outstanding   // high-water mark of in-flight MSHRs
);
    //------------------------------------------------------------------------
    // Directory
    //------------------------------------------------------------------------
    reg               valid_arr [0:SLOTS-1];   // data resident
    reg  [ID_W-1:0]   tag_arr   [0:SLOTS-1];   // resident expert id
    reg  [SLOT_W-1:0] rank      [0:SLOTS-1];   // recency (0 = MRU)
    reg               resv_arr  [0:SLOTS-1];   // reserved by an in-flight MSHR

    //------------------------------------------------------------------------
    // MSHR file
    //------------------------------------------------------------------------
    reg                   m_valid [0:MSHR_DEPTH-1];  // entry occupied
    reg  [ID_W-1:0]       m_id    [0:MSHR_DEPTH-1];  // expert being fetched
    reg  [SLOT_W-1:0]     m_slot  [0:MSHR_DEPTH-1];  // reserved victim slot
    reg                   m_done  [0:MSHR_DEPTH-1];  // data arrived, response owed
    reg                   m_prim  [0:MSHR_DEPTH-1];  // 1 = issued the fetch (primary)
    reg                   m_sent  [0:MSHR_DEPTH-1];  // primary's flash req accepted

    integer k, j;

    localparam [SLOT_W-1:0] LRU_POS = SLOT_W'(SLOTS-1);

    //------------------------------------------------------------------------
    // Lookup: a RESERVED slot is NOT a hit (hazard 3) -- its data has not landed.
    //------------------------------------------------------------------------
    reg               lookup_hit;
    reg  [SLOT_W-1:0] lookup_slot;
    always @* begin
        lookup_hit  = 1'b0;
        lookup_slot = {SLOT_W{1'b0}};
        for (k = 0; k < SLOTS; k = k + 1)
            if (valid_arr[k] && !resv_arr[k] && (tag_arr[k] == req_expert_id)) begin
                lookup_hit  = 1'b1;
                lookup_slot = k[SLOT_W-1:0];
            end
    end

    //------------------------------------------------------------------------
    // MSHR match: is this id already in flight?  (hazard 2 -- merge, don't refetch)
    //------------------------------------------------------------------------
    reg               mshr_match;
    reg  [SLOT_W-1:0] mshr_match_slot;
    always @* begin
        mshr_match      = 1'b0;
        mshr_match_slot = {SLOT_W{1'b0}};
        for (k = 0; k < MSHR_DEPTH; k = k + 1)
            if (m_valid[k] && (m_id[k] == req_expert_id)) begin
                mshr_match      = 1'b1;
                mshr_match_slot = m_slot[k];
            end
    end

    //------------------------------------------------------------------------
    // Free MSHR entry (lowest index)
    //------------------------------------------------------------------------
    reg               have_free;
    reg  [TAG_W-1:0]  free_tag;
    always @* begin
        have_free = 1'b0;
        free_tag  = {TAG_W{1'b0}};
        for (k = MSHR_DEPTH-1; k >= 0; k = k - 1)
            if (!m_valid[k]) begin
                have_free = 1'b1;
                free_tag  = k[TAG_W-1:0];
            end
    end

    //------------------------------------------------------------------------
    // Slot busy: ANY live MSHR entry targets this slot.
    //   NOTE this is BROADER than resv_arr, and the difference is load-bearing.
    //   resv_arr[s] means "data has not arrived" (cleared at install, so the slot
    //   can start answering hits).  But an entry stays live from install until it
    //   has RESPONDED, and a merged entry may still owe a response after the
    //   primary installed.  If victim selection only excluded resv_arr, a fresh
    //   miss could re-victimise the slot in that window and the pending response
    //   would name a slot that no longer holds its expert.  (Found by
    //   test/expert_cache_mshr_tb.v's response checker, not by inspection.)
    //------------------------------------------------------------------------
    reg slot_busy [0:SLOTS-1];
    always @* begin
        for (k = 0; k < SLOTS; k = k + 1) slot_busy[k] = 1'b0;
        for (k = 0; k < MSHR_DEPTH; k = k + 1)
            if (m_valid[k]) slot_busy[m_slot[k]] = 1'b1;
    end

    //------------------------------------------------------------------------
    // Victim select -- slots owned by a live MSHR are excluded (hazard 1).
    //   Injection -DINJECT_NORESV drops the exclusion so two concurrent misses
    //   can pick the same slot: the TB's directory check must catch it.
    //------------------------------------------------------------------------
`ifdef INJECT_NORESV
    `define RESV_EXCL(i) 1'b1
`else
    `define RESV_EXCL(i) (!slot_busy[i])
`endif
    reg               have_invalid;
    reg  [SLOT_W-1:0] invalid_slot;
    reg               have_lru;
    reg  [SLOT_W-1:0] lru_slot;
    reg  [SLOT_W-1:0] best_rank;
    always @* begin
        have_invalid = 1'b0;
        invalid_slot = {SLOT_W{1'b0}};
        // lowest-index invalid AND unreserved wins (scan high->low, last wins)
        for (k = SLOTS-1; k >= 0; k = k - 1)
            if (!valid_arr[k] && `RESV_EXCL(k)) begin
                have_invalid = 1'b1;
                invalid_slot = k[SLOT_W-1:0];
            end
        // LRU among UNRESERVED slots: the largest rank that is not reserved
        have_lru  = 1'b0;
        lru_slot  = {SLOT_W{1'b0}};
        best_rank = {SLOT_W{1'b0}};
        for (k = 0; k < SLOTS; k = k + 1)
            if (`RESV_EXCL(k) && (!have_lru || (rank[k] > best_rank))) begin
                have_lru  = 1'b1;
                lru_slot  = k[SLOT_W-1:0];
                best_rank = rank[k];
            end
    end
    wire              have_victim = have_invalid || have_lru;
    wire [SLOT_W-1:0] victim_slot = have_invalid ? invalid_slot : lru_slot;

    //------------------------------------------------------------------------
    // Acceptance: a HIT always lands; a MERGE needs an MSHR entry; a fresh MISS
    // needs an MSHR entry AND an unreserved victim.
    //------------------------------------------------------------------------
    //   RANK HAZARD: a demand HIT and a completing INSTALL both move-to-front the
    //   recency permutation.  Two touches in one cycle would corrupt {rank} (the
    //   second task call's non-blocking writes overwrite the first's).  The
    //   install cannot be deferred (the data has arrived), so a HIT is not
    //   accepted on an install cycle -- one cycle of back-pressure via req_ready,
    //   visible to the caller, rare in practice.  Misses/merges do not touch rank.
    wire install_now  = flash_done;
    wire accept_hit   = req_valid && lookup_hit && !install_now;
    wire accept_merge = req_valid && !lookup_hit && mshr_match && have_free;
    wire accept_miss  = req_valid && !lookup_hit && !mshr_match && have_free && have_victim;
    assign req_ready  = lookup_hit ? !install_now
                                   : (mshr_match ? have_free : (have_free && have_victim));

    //------------------------------------------------------------------------
    // Completion scheduler: pick one done MSHR to answer (lowest index).
    //------------------------------------------------------------------------
    reg              have_done;
    reg [TAG_W-1:0]  done_tag;
    always @* begin
        have_done = 1'b0;
        done_tag  = {TAG_W{1'b0}};
        for (k = MSHR_DEPTH-1; k >= 0; k = k - 1)
            if (m_valid[k] && m_done[k]) begin
                have_done = 1'b1;
                done_tag  = k[TAG_W-1:0];
            end
    end

    //------------------------------------------------------------------------
    // Next un-sent primary (its flash request still needs issuing)
    //------------------------------------------------------------------------
    reg              have_tosend;
    reg [TAG_W-1:0]  tosend_tag;
    always @* begin
        have_tosend = 1'b0;
        tosend_tag  = {TAG_W{1'b0}};
        for (k = MSHR_DEPTH-1; k >= 0; k = k - 1)
            if (m_valid[k] && m_prim[k] && !m_sent[k]) begin
                have_tosend = 1'b1;
                tosend_tag  = k[TAG_W-1:0];
            end
    end

    // outstanding count (for the high-water stat)
    reg [31:0] outstanding_n;
    always @* begin
        outstanding_n = 32'd0;
        for (k = 0; k < MSHR_DEPTH; k = k + 1)
            if (m_valid[k]) outstanding_n = outstanding_n + 32'd1;
    end

    //------------------------------------------------------------------------
    // Move-to-front LRU touch (same form as the committed cache)
    //------------------------------------------------------------------------
    task lru_touch(input [SLOT_W-1:0] s);
        begin
            for (j = 0; j < SLOTS; j = j + 1)
                if (rank[j] < rank[s]) rank[j] <= rank[j] + 1'b1;
            rank[s] <= {SLOT_W{1'b0}};
        end
    endtask

    //------------------------------------------------------------------------
    // SEQUENTIAL
    //------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            resp_valid        <= 1'b0;
            resp_hit          <= 1'b0;
            resp_slot         <= {SLOT_W{1'b0}};
            resp_expert_id    <= {ID_W{1'b0}};
            flash_req         <= 1'b0;
            flash_expert_id   <= {ID_W{1'b0}};
            flash_tag         <= {TAG_W{1'b0}};
            hit_count         <= 32'd0;
            miss_count        <= 32'd0;
            fetch_count       <= 32'd0;
            merge_count       <= 32'd0;
            full_stall_cycles <= 32'd0;
            max_outstanding   <= 32'd0;
            for (k = 0; k < SLOTS; k = k + 1) begin
                valid_arr[k] <= 1'b0;
                tag_arr[k]   <= {ID_W{1'b0}};
                rank[k]      <= k[SLOT_W-1:0];
                resv_arr[k]  <= 1'b0;
            end
            for (k = 0; k < MSHR_DEPTH; k = k + 1) begin
                m_valid[k] <= 1'b0;
                m_id[k]    <= {ID_W{1'b0}};
                m_slot[k]  <= {SLOT_W{1'b0}};
                m_done[k]  <= 1'b0;
                m_prim[k]  <= 1'b0;
                m_sent[k]  <= 1'b0;
            end
        end else begin
            resp_valid <= 1'b0;                 // response is a 1-cycle pulse

            if (req_valid && !req_ready)
                full_stall_cycles <= full_stall_cycles + 32'd1;
            if (outstanding_n > max_outstanding)
                max_outstanding <= outstanding_n;

            //================================================================
            // 1) ACCEPT a request
            //================================================================
            if (accept_hit) begin
                // HIT: answer immediately, touch recency.
                resp_valid     <= 1'b1;
                resp_hit       <= 1'b1;
                resp_slot      <= lookup_slot;
                resp_expert_id <= req_expert_id;
                hit_count      <= hit_count + 32'd1;
                lru_touch(lookup_slot);
            end else if (accept_merge) begin
                // MERGE (hazard 2): bind to the in-flight primary's slot, NO fetch.
                m_valid[free_tag] <= 1'b1;
                m_id[free_tag]    <= req_expert_id;
                m_slot[free_tag]  <= mshr_match_slot;
                m_done[free_tag]  <= 1'b0;
`ifdef INJECT_NOMERGE
                // INJECTION: treat it as a fresh fetch (duplicate flash traffic).
                m_prim[free_tag]  <= 1'b1;
                m_sent[free_tag]  <= 1'b0;
`else
                m_prim[free_tag]  <= 1'b0;      // not the fetcher
                m_sent[free_tag]  <= 1'b1;      // nothing to send
`endif
                miss_count        <= miss_count + 32'd1;
                merge_count       <= merge_count + 32'd1;
            end else if (accept_miss) begin
                // FRESH MISS: allocate, RESERVE the victim (hazard 1), fetch.
                m_valid[free_tag] <= 1'b1;
                m_id[free_tag]    <= req_expert_id;
                m_slot[free_tag]  <= victim_slot;
                m_done[free_tag]  <= 1'b0;
                m_prim[free_tag]  <= 1'b1;
                m_sent[free_tag]  <= 1'b0;
                resv_arr[victim_slot] <= 1'b1;
                valid_arr[victim_slot] <= 1'b0;   // its old data is gone from now on
                miss_count        <= miss_count + 32'd1;
            end

            //================================================================
            // 2) ISSUE a flash fetch for the oldest un-sent primary
            //================================================================
            flash_req <= 1'b0;
            if (have_tosend && flash_ready) begin
                flash_req       <= 1'b1;
                flash_expert_id <= m_id[tosend_tag];
                flash_tag       <= tosend_tag;
                m_sent[tosend_tag] <= 1'b1;
                fetch_count     <= fetch_count + 32'd1;
            end

            //================================================================
            // 3) COMPLETE: install on the primary's tagged done; wake merges
            //================================================================
            if (flash_done) begin
                // install into the reserved slot
                valid_arr[m_slot[flash_done_tag]] <= 1'b1;
                tag_arr  [m_slot[flash_done_tag]] <= m_id[flash_done_tag];
                resv_arr [m_slot[flash_done_tag]] <= 1'b0;
                lru_touch(m_slot[flash_done_tag]);
                m_done[flash_done_tag] <= 1'b1;
                // any MERGED entry waiting on the same slot is now satisfiable
                for (k = 0; k < MSHR_DEPTH; k = k + 1)
                    if (m_valid[k] && !m_prim[k] && (m_slot[k] == m_slot[flash_done_tag]))
                        m_done[k] <= 1'b1;
            end

            //================================================================
            // 4) RESPOND for one done MSHR (a hit this cycle already answered)
            //================================================================
            if (have_done && !accept_hit) begin
                resp_valid     <= 1'b1;
                resp_hit       <= 1'b0;
                resp_slot      <= m_slot[done_tag];
                resp_expert_id <= m_id[done_tag];
                m_valid[done_tag] <= 1'b0;
                m_done[done_tag]  <= 1'b0;
                m_prim[done_tag]  <= 1'b0;
                m_sent[done_tag]  <= 1'b0;
            end
        end
    end
endmodule
/* verilator lint_on DECLFILENAME */
