// Registered probe exposing f32_pkg ops for TB fuzzing against numpy.
module f32_probe (
    input  logic clk,
    input  logic [31:0] a,
    input  logic [31:0] b,
    input  logic        sel,   // 0: mul, 1: add
    output logic [31:0] y
);
    import f32_pkg::*;
    always_ff @(posedge clk)
        y <= sel ? f32_add(a, b) : f32_mul(a, b);
endmodule