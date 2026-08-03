// l3_top_mark24.v -- INJECTION-ONLY top: instantiate the L3 board top with the
//   loopback marker FORCED back to its default position (24).  At the L3 fitted
//   geometry the fw loopback key needs 25 bits, so glm_q4k_system's elaboration
//   guard MUST reject this build with its $error.  This file is compiled only by
//   the l3-elab liveness leg in the Makefile; if it ever elaborates cleanly, the
//   guard -- and the whole LB_MARKER_LSB mechanism -- has stopped constraining
//   anything and the gate says so loudly.
module l3_top_mark24;
    l3_top #(.LB_MARKER_LSB(24)) u_top (
        .host_clk(1'b0), .host_rst(1'b1), .core_clk(1'b0), .core_rst(1'b1),
        .uart_rx(1'b1), .uart_tx(),
        .spi_cs_n(), .spi_sclk(), .spi_mosi(), .spi_miso(1'b0),
        .boot_done_led(), .boot_fail_led(),
        .m_axi_arid(), .m_axi_araddr(), .m_axi_arlen(), .m_axi_arsize(),
        .m_axi_arburst(), .m_axi_arvalid(), .m_axi_arready(1'b0),
        .m_axi_rid({9{1'b0}}), .m_axi_rdata({256{1'b0}}), .m_axi_rresp(2'b00),
        .m_axi_rlast(1'b0), .m_axi_rvalid(1'b0), .m_axi_rready(),
        .m_axi_awaddr(), .m_axi_awlen(), .m_axi_awsize(), .m_axi_awburst(),
        .m_axi_awvalid(), .m_axi_awready(1'b0),
        .m_axi_wdata(), .m_axi_wstrb(), .m_axi_wlast(), .m_axi_wvalid(),
        .m_axi_wready(1'b0),
        .m_axi_bresp(2'b00), .m_axi_bvalid(1'b0), .m_axi_bready()
    );
endmodule
