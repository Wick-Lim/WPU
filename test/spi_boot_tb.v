`timescale 1ns/1ps
//============================================================================
// test/spi_boot_tb.v -- boot_loader + spi_flash_reader INTEGRATION (`make spi-boot`)
//
// WHAT IS PROVEN
//   The L3 storage path is exactly one job: boot_loader's power-up Flash->DDR
//   copy (at RESIDENT=1 runtime decode never touches Flash for weights).  This TB
//   closes that path end-to-end IN SIMULATION:
//
//     boot_loader  --flash_req/addr-->  spi_flash_reader  --SPI mode-0 pins-->
//     behavioural SPI-NOR slave (serves a known byte image)
//     ... and boot_loader's ddr_we/ddr_addr/ddr_wdata land in a DDR sink.
//
//   Checks:
//     B1  boot_loader raises done and not boot_fail.
//     B2  EVERY DDR word equals the flash image bytes at the segment mapping --
//         byte-for-byte, both segments, MSB-first word assembly.
//     B3  words_done == total words copied.
//     B4  the SPI slave saw only legal 0x03 READ commands.
//
// INJECTION (`make spi-boot` requires FAILURE)
//   -DINJ_SPI_BYTESWAP assembles each word with its halves swapped inside the
//   reader.  B2 MUST fail -- proving the image comparison constrains byte order,
//   not merely "data arrived".
//
// NOT PROVEN HERE (board-only, stated): real NOR part timing, quad mode, power-up
//   delays, the physical pins.  This is the contract-composition proof only.
//============================================================================
module spi_boot_tb;

    localparam integer ADDR_W = 32;
    localparam integer DATA_W = 64;
    localparam integer NBYTES = DATA_W/8;
    localparam integer SEG_MAX = 4;
    localparam integer LEN_W  = 16;
    localparam integer IMG_BYTES = 8192;

    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst;

    // ---- boot_loader <-> reader ----
    wire              f_req;
    wire [ADDR_W-1:0] f_addr;
    wire              f_ready, f_rvalid;
    wire [DATA_W-1:0] f_rdata;
    // ---- SPI pins ----
    wire cs_n, sclk, mosi;
    wire miso;
    // ---- DDR sink ----
    wire              ddr_we;
    wire [ADDR_W-1:0] ddr_addr;
    wire [DATA_W-1:0] ddr_wdata;
    wire              busy, done, boot_fail;
    wire [2:0]        err_code;
    wire [31:0]       words_done;

    // two segments: {flash word base, ddr word base, length in words}
    localparam integer S0_F = 0,   S0_D = 32'h100, S0_L = 32;
    localparam integer S1_F = 64,  S1_D = 32'h200, S1_L = 16;

    reg start;
    boot_loader #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .SEG_MAX(SEG_MAX),
                  .BURST(8), .LEN_W(LEN_W), .INTEGRITY(0)) u_boot (
        .clk(clk), .rst(rst), .start(start),
        .seg_count(3'd2),
        .seg_flash_base({{(SEG_MAX-2)*ADDR_W{1'b0}}, 32'(S1_F), 32'(S0_F)}),
        .seg_ddr_base  ({{(SEG_MAX-2)*ADDR_W{1'b0}}, 32'(S1_D), 32'(S0_D)}),
        .seg_len       ({{(SEG_MAX-2)*LEN_W{1'b0}},  16'(S1_L), 16'(S0_L)}),
        .mf_magic(32'h4D4F_444C), .mf_version(16'd1),
        .mf_len(32'd0), .mf_crc(32'd0),
        .flash_req(f_req), .flash_addr(f_addr), .flash_ready(f_ready),
        .flash_rvalid(f_rvalid), .flash_rdata(f_rdata),
        .ddr_we(ddr_we), .ddr_addr(ddr_addr), .ddr_wdata(ddr_wdata),
        .ddr_ready(1'b1),
        .busy(busy), .done(done), .words_done(words_done),
        .boot_fail(boot_fail), .err_code(err_code));

    spi_flash_reader #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) u_rd (
        .clk(clk), .rst(rst),
        .flash_req(f_req), .flash_addr(f_addr), .flash_ready(f_ready),
        .flash_rvalid(f_rvalid), .flash_rdata(f_rdata),
        .spi_cs_n(cs_n), .spi_sclk(sclk), .spi_mosi(mosi), .spi_miso(miso));

    // ================= behavioural SPI-NOR slave (mode 0) ====================
    //   Samples MOSI on SCLK rising; presents MISO on SCLK falling.  Serves the
    //   byte image `img`, MSB first, after an 8-bit command + 24-bit address.
    reg [7:0]  img [0:IMG_BYTES-1];
    reg [31:0] hdr_sh;
    integer    hdr_n;         // header bits captured
    integer    dbit_n;        // data bit index within the current byte
    integer    baddr;         // current byte address
    reg        miso_r;
    integer    bad_cmds;      // B4: non-0x03 commands seen
    assign miso = miso_r;

    always @(negedge cs_n) begin
        hdr_n = 0; dbit_n = 0; baddr = 0; hdr_sh = 32'd0;
    end
    always @(posedge sclk) if (!cs_n) begin
        if (hdr_n < 32) begin
            hdr_sh = {hdr_sh[30:0], mosi};
            hdr_n  = hdr_n + 1;
            if (hdr_n == 32) begin
                if (hdr_sh[31:24] !== 8'h03) bad_cmds = bad_cmds + 1;
                baddr  = hdr_sh[23:0];
                dbit_n = 0;
            end
        end
    end
    always @(negedge sclk) if (!cs_n && hdr_n == 32) begin
        miso_r = img[baddr[12:0]][7 - dbit_n];
        dbit_n = dbit_n + 1;
        if (dbit_n == 8) begin dbit_n = 0; baddr = baddr + 1; end
    end

    // ================= DDR sink ==============================================
    reg [DATA_W-1:0] ddr_mem [0:1023];
    always @(posedge clk) if (ddr_we) ddr_mem[ddr_addr[9:0]] <= ddr_wdata;

    // ================= checks ================================================
    integer errors, tests, i, b;
    reg [DATA_W-1:0] exp_w;
    task chk(input cond, input [8*72-1:0] name);
        begin tests = tests + 1;
              if (!cond) begin errors = errors + 1; $display("FAIL: %0s", name); end end
    endtask

    task check_seg(input integer fbase, input integer dbase, input integer len);
        begin
            for (i = 0; i < len; i = i + 1) begin
                for (b = 0; b < NBYTES; b = b + 1)
                    exp_w[DATA_W-1-8*b -: 8] = img[(fbase+i)*NBYTES + b];
                tests = tests + 1;
                if (ddr_mem[(dbase+i) & 1023] !== exp_w) begin
                    if (errors < 6)
                        $display("FAIL[B2]: ddr[%0h] = %h, image says %h",
                                 dbase+i, ddr_mem[(dbase+i) & 1023], exp_w);
                    errors = errors + 1;
                end
            end
        end
    endtask

    initial begin
        errors = 0; tests = 0; bad_cmds = 0; miso_r = 1'b0; start = 1'b0;
        // deterministic image: every byte is a mix of its address bits
        for (i = 0; i < IMG_BYTES; i = i + 1)
            img[i] = i[7:0] ^ {i[10:8], 5'h15} ^ 8'hA5;
        for (i = 0; i < 1024; i = i + 1) ddr_mem[i] = {DATA_W{1'bx}};

        rst = 1'b1; repeat (5) @(negedge clk); rst = 1'b0;
        repeat (2) @(negedge clk);
        start = 1'b1; @(negedge clk); start = 1'b0;

        wait (done || boot_fail);
        repeat (10) @(negedge clk);

        chk(done && !boot_fail, "B1: boot completed without boot_fail");
        check_seg(S0_F, S0_D, S0_L);
        check_seg(S1_F, S1_D, S1_L);
        chk(words_done == S0_L + S1_L, "B3: words_done == total copied");
        chk(bad_cmds == 0, "B4: the SPI slave saw only 0x03 READ commands");

        if (errors != 0) begin
            $display("FAILED: %0d error(s) across %0d checks", errors, tests);
            $fatal(1, "spi_boot_tb had mismatches");
        end
        $display("ALL %0d TESTS PASSED  (boot_loader -> spi_flash_reader -> SPI-NOR: %0d words copied to DDR byte-for-byte over real mode-0 SPI waveforms)",
                 tests, S0_L + S1_L);
        $finish;
    end

    initial begin
        #40000000;
        $display("FAIL: global timeout (boot never finished)");
        $fatal(1, "timeout");
    end
endmodule
