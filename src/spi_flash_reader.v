// ============================================================================
// spi_flash_reader.v -- boot_loader's Flash port  ->  SPI-NOR flash (L3 S4)
//
//   ROLE IN THE L3 BRING-UP
//   At RESIDENT=1 the runtime decode NEVER touches the Flash path for weights --
//   every expert refill is a real banked ddr5_xbar read (served through
//   ddr4_mig_shim).  What remains on the storage side is exactly ONE job: the
//   power-up Flash->DDR copy that boot_loader already sequences.  boot_loader's
//   Flash face is a simple in-order read contract:
//
//       flash_req/flash_addr  (request strobe, held until flash_ready)
//       flash_rvalid/flash_rdata  (returned words, IN ORDER)
//
//   This module implements that contract on top of a standard SPI-NOR READ
//   (command 0x03: 8 command bits, 24 address bits, then data out on MISO,
//   MSB first) -- the one read mode every NOR flash supports and the same part
//   an FPGA board already carries for configuration.  Single-lane on purpose:
//   the L3 image is a compact-config resident set (MBs, not GBs), the boot copy
//   is a one-time bulk read, and at SCLK = clk/2 it finishes in seconds.  Quad
//   I/O is a later optimisation, not a correctness need.
//
//   WORD ASSEMBLY.  boot_loader consumes DATA_W-bit words; the flash returns
//   bytes.  Words are assembled MSB-first (byte 0 of the image is the top byte
//   of word 0) and presented on flash_rvalid for exactly one cycle each.
//   flash_addr is a WORD address; the SPI byte address is addr * (DATA_W/8).
//
//   BURST CONTRACT.  boot_loader may hold flash_req with a new address as soon
//   as flash_ready is seen (BURST reads in flight through its skid FIFO).  This
//   reader accepts ONE request at a time (ready drops while busy) -- legal,
//   because flash_ready is a handshake, not a promise of pipelining; the
//   loader's FIFO simply never fills.  Boot time is not the L3 bottleneck.
//
//   NOT VERIFIED HERE (stated, not pretended): a real flash part's timing
//   (max SCLK, tSLCH), power-up delays, quad-enable bits, and anything about a
//   board.  Verified in simulation against a behavioural SPI-NOR model that
//   serves a known image, byte-for-byte.
// ============================================================================
`default_nettype none

module spi_flash_reader #(
    parameter integer ADDR_W = 32,   // WORD address width (boot_loader side)
    parameter integer DATA_W = 64    // word width; must be a multiple of 8
)(
    input  wire               clk,
    input  wire               rst,

    // ---- boot_loader Flash face (this module is the "flash") ----
    input  wire               flash_req,
    input  wire [ADDR_W-1:0]  flash_addr,
    output wire               flash_ready,
    output reg                flash_rvalid,
    output reg  [DATA_W-1:0]  flash_rdata,

    // ---- SPI pins ----
    output reg                spi_cs_n,
    output reg                spi_sclk,
    output reg                spi_mosi,
    input  wire               spi_miso
);

    localparam integer NBYTES = DATA_W / 8;
    // 8 command bits + 24 address bits shifted out, then NBYTES*8 data bits in
    localparam integer HDR_BITS  = 32;
    localparam [7:0]   CMD_READ  = 8'h03;

    localparam [1:0] S_IDLE = 2'd0, S_HDR = 2'd1, S_DATA = 2'd2, S_DONE = 2'd3;
    reg [1:0]  st;

    reg [31:0] hdr_sh;                    // {CMD, addr[23:0]} shifting out
    reg [5:0]  bit_cnt;                   // bits remaining in the current phase
    reg [$clog2(DATA_W+1)-1:0] dbit;      // data bits captured
    reg [DATA_W-1:0] dsh;                 // data shift register (MSB first)
    reg        phase;                     // 0: about to drive falling edge work
                                          // (sclk toggled inside the FSM below)

    assign flash_ready = (st == S_IDLE);

    always @(posedge clk) begin
        if (rst) begin
            st          <= S_IDLE;
            spi_cs_n    <= 1'b1;
            spi_sclk    <= 1'b0;
            spi_mosi    <= 1'b0;
            flash_rvalid<= 1'b0;
            flash_rdata <= {DATA_W{1'b0}};
            hdr_sh      <= 32'd0;
            bit_cnt     <= 6'd0;
            dbit        <= {$clog2(DATA_W+1){1'b0}};
            dsh         <= {DATA_W{1'b0}};
            phase       <= 1'b0;
        end else begin
            flash_rvalid <= 1'b0;
            case (st)
            S_IDLE: begin
                spi_cs_n <= 1'b1;
                spi_sclk <= 1'b0;
                if (flash_req) begin
                    // WORD address -> SPI byte address (24-bit NOR address space)
                    hdr_sh   <= {CMD_READ, flash_addr[23-$clog2(NBYTES):0],
                                 {$clog2(NBYTES){1'b0}}};
                    bit_cnt  <= HDR_BITS[5:0];
                    dbit     <= {$clog2(DATA_W+1){1'b0}};
                    spi_cs_n <= 1'b0;
                    phase    <= 1'b0;
                    st       <= S_HDR;
                end
            end
            // SPI mode 0: MOSI changes while SCLK low, slave samples on rising
            // edge; MISO is sampled by us on the falling edge after a rise.
            S_HDR: begin
                if (!phase) begin
                    spi_mosi <= hdr_sh[31];       // present bit with SCLK low
                    spi_sclk <= 1'b1;             // rising edge: slave samples
                    phase    <= 1'b1;
                end else begin
                    spi_sclk <= 1'b0;             // falling edge
                    hdr_sh   <= {hdr_sh[30:0], 1'b0};
                    bit_cnt  <= bit_cnt - 6'd1;
                    phase    <= 1'b0;
                    if (bit_cnt == 6'd1) st <= S_DATA;
                end
            end
            S_DATA: begin
                if (!phase) begin
                    spi_sclk <= 1'b1;             // rising edge: slave presents
                    phase    <= 1'b1;
                end else begin
                    spi_sclk <= 1'b0;
                    dsh      <= {dsh[DATA_W-2:0], spi_miso};  // sample on fall
                    dbit     <= dbit + 1'b1;
                    phase    <= 1'b0;
                    if (dbit == DATA_W[$clog2(DATA_W+1)-1:0] - 1'b1) st <= S_DONE;
                end
            end
            S_DONE: begin
                spi_cs_n     <= 1'b1;
                spi_sclk     <= 1'b0;
`ifdef INJ_SPI_BYTESWAP
                // INJECTION (never a normal build): assemble the word with the
                //   halves swapped.  Every
                //   multi-byte word the boot image carries is then wrong, and the TB's
                //   byte-for-byte image comparison MUST fail -- proving the
                //   comparison constrains assembly order, not just "data arrived".
                flash_rdata  <= {dsh[DATA_W/2-1:0], dsh[DATA_W-1:DATA_W/2]};
`else
                flash_rdata  <= dsh;
`endif
                flash_rvalid <= 1'b1;
                st           <= S_IDLE;
            end
            endcase
        end
    end

endmodule
`default_nettype wire
