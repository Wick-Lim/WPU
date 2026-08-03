`timescale 1ns/1ps
//============================================================================
// test/axi_boot_writer_tb.v -- gate for src/axi_boot_writer.v  (`make boot-writer`)
//
// WHAT IS PROVEN
//   The writer converts boot_loader's combinational ddr_we/ddr_addr/ddr_wdata
//   into AXI4 write channels whose payload is STABLE while AWVALID/WVALID are
//   held.  The requester side here is HOSTILE on purpose: it holds ddr_we and
//   CHANGES {addr,wdata} every cycle while ddr_ready is low -- the same
//   retarget-on-refusal waveform the read side measured on the real system
//   (unstable_cyc=60).  Against an AXI write slave that refuses roughly half
//   the time on BOTH the AW and W channels and delays B, the TB asserts:
//     W1  AWADDR never moves while AWVALID is held (audited in the TB, and the
//         DUT's own `ifndef YOSYS assertion $fatals independently);
//     W2  every ACCEPTED write lands in the slave's memory at its byte address
//         with its ORIGINAL data -- not whatever the requester showed later;
//     W3  no accepted write is lost and none is duplicated (count equality);
//     W4  dbg_writes matches and no BRESP error was returned.
//
// INJECTION (`make boot-writer` requires FAILURE)
//   -DINJ_BWR_NOSLOT drives AWADDR straight from the churning requester instead
//   of the registered slot.  W1/W2 MUST fail -- otherwise the slot constrains
//   nothing and the pass above would be vacuous.
//============================================================================
module axi_boot_writer_tb;

    localparam integer ADDR_W = 32;
    localparam integer DATA_W = 64;
    localparam integer NWRITES = 400;

    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst;

    reg                ddr_we;
    reg  [ADDR_W-1:0]  ddr_addr;
    reg  [DATA_W-1:0]  ddr_wdata;
    wire               ddr_ready;

    wire [31:0] awaddr;  wire [7:0] awlen;  wire [2:0] awsize;
    wire [1:0]  awburst; wire awvalid;      reg  awready;
    wire [DATA_W-1:0] wdata; wire [DATA_W/8-1:0] wstrb;
    wire wlast, wvalid;  reg wready;
    reg  [1:0] bresp;    reg bvalid;  wire bready;
    wire [31:0] dbg_w, dbg_e;

    axi_boot_writer #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) dut (
        .clk(clk), .rst(rst),
        .ddr_we(ddr_we), .ddr_addr(ddr_addr), .ddr_wdata(ddr_wdata),
        .ddr_ready(ddr_ready),
        .m_axi_awaddr(awaddr), .m_axi_awlen(awlen), .m_axi_awsize(awsize),
        .m_axi_awburst(awburst), .m_axi_awvalid(awvalid), .m_axi_awready(awready),
        .m_axi_wdata(wdata), .m_axi_wstrb(wstrb), .m_axi_wlast(wlast),
        .m_axi_wvalid(wvalid), .m_axi_wready(wready),
        .m_axi_bresp(bresp), .m_axi_bvalid(bvalid), .m_axi_bready(bready),
        .dbg_writes(dbg_w), .dbg_bresp_err(dbg_e));

    integer errors, tests;
    task chk(input cond, input [8*72-1:0] name);
        begin tests = tests + 1;
              if (!cond) begin errors = errors + 1; $display("FAIL: %0s", name); end end
    endtask

    // ---- what the writer ACCEPTED (the contract it must honour) --------------
    integer n_acc;
    reg [ADDR_W-1:0] acc_addr [0:NWRITES-1];
    reg [DATA_W-1:0] acc_data [0:NWRITES-1];

    // ---- hostile requester: churn a held payload every cycle -----------------
    integer lfsr_s, churn;
    always @(posedge clk) if (!rst) begin
        lfsr_s = {lfsr_s[30:0], lfsr_s[31]^lfsr_s[21]^lfsr_s[1]^lfsr_s[0]};
        if (n_acc < NWRITES) begin
            ddr_we    <= 1'b1;
            ddr_addr  <= {12'h0, lfsr_s[19:0]};
            ddr_wdata <= {lfsr_s[31:0], ~lfsr_s[31:0]};
            if (ddr_we && !ddr_ready) churn = churn + 1;
        end else ddr_we <= 1'b0;
    end
    always @(posedge clk) if (!rst && ddr_we && ddr_ready && n_acc < NWRITES) begin
        acc_addr[n_acc] = ddr_addr;
        acc_data[n_acc] = ddr_wdata;
        n_acc = n_acc + 1;
    end

    // ---- W1 audit: AW stability while held ------------------------------------
    reg [31:0] aw_q;  reg aw_h;
    always @(posedge clk) if (!rst) begin
        if (awvalid && !awready) begin
            if (aw_h && (awaddr !== aw_q)) begin
                if (errors < 5) $display("FAIL[W1]: AWADDR moved while held (%h -> %h)", aw_q, awaddr);
                errors = errors + 1;
            end
            aw_q <= awaddr; aw_h <= 1'b1;
        end else aw_h <= 1'b0;
    end

    // ---- AXI write slave: refuse ~half on AW and W independently, delay B -----
    reg [31:0] lfsr;
    reg [DATA_W-1:0] mem [0:(1<<20)-1];        // byte addr >> 3 indexes
    reg [31:0] pend_addr;  reg pend_aw, pend_w;
    reg [DATA_W-1:0] pend_data;
    integer b_delay;
    always @(posedge clk) begin
        if (rst) begin
            awready <= 1'b0; wready <= 1'b0; bvalid <= 1'b0; bresp <= 2'b00;
            lfsr = 32'hCAFE_F00D; pend_aw = 0; pend_w = 0; b_delay = 0;
        end else begin
            lfsr = {lfsr[30:0], lfsr[31]^lfsr[21]^lfsr[1]^lfsr[0]};
            awready <= lfsr[2];
            wready  <= lfsr[7];
            if (awvalid && awready) begin pend_addr = awaddr; pend_aw = 1; end
            if (wvalid  && wready ) begin pend_data = wdata;  pend_w  = 1; end
            if (bvalid && bready) bvalid <= 1'b0;
            if (pend_aw && pend_w && !bvalid && b_delay == 0) b_delay = 2 + lfsr[5:4];
            if (b_delay > 1) b_delay = b_delay - 1;
            else if (b_delay == 1) begin
                mem[pend_addr[22:3]] <= pend_data;      // commit at byte addr
                bvalid <= 1'b1; bresp <= 2'b00;
                pend_aw = 0; pend_w = 0; b_delay = 0;
            end
        end
    end

    integer i;
    initial begin
        errors = 0; tests = 0; n_acc = 0; churn = 0;
        ddr_we = 0; ddr_addr = 0; ddr_wdata = 0; lfsr_s = 32'h1357_9BDF;
        aw_h = 0;
        rst = 1'b1; repeat (5) @(negedge clk); rst = 1'b0;

        wait (n_acc == NWRITES);
        // drain: all accepted writes must retire
        repeat (4000) @(posedge clk);

        $display("  [MEASURED] churn cycles=%0d, accepted=%0d, AXI writes retired=%0d",
                 churn, n_acc, dbg_w);
        chk(churn > 200,      "the requester really churned held payloads (else W1 is vacuous)");
        chk(dbg_w == NWRITES, "W3: every accepted write retired exactly once");
        chk(dbg_e == 0,       "W4: no BRESP error");

        // W2: memory holds the ORIGINAL accepted data at the ORIGINAL byte addr
        for (i = 0; i < NWRITES; i = i + 1) begin
            tests = tests + 1;
            if (mem[{acc_addr[i], 3'b000} >> 3] !== acc_data[i]) begin
                if (errors < 6)
                    $display("FAIL[W2]: word addr %h holds %h, accepted data was %h",
                             acc_addr[i], mem[{acc_addr[i], 3'b000} >> 3], acc_data[i]);
                errors = errors + 1;
            end
        end

        if (errors != 0) begin
            $display("FAILED: %0d error(s) across %0d checks", errors, tests);
            $fatal(1, "axi_boot_writer_tb had mismatches");
        end
        $display("ALL %0d TESTS PASSED  (axi_boot_writer: %0d writes retired intact under a requester that retargets refused writes every cycle and a slave that refuses ~half of AW and W beats)",
                 tests, NWRITES);
        $finish;
    end

    initial begin
        #4000000;
        $display("FAIL: global timeout");
        $fatal(1, "timeout");
    end
endmodule
