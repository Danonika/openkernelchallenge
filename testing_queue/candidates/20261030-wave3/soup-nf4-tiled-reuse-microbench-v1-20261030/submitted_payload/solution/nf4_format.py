"""Public NF4 FP32 format constants; original selector, no vendor kernel."""
import triton
import triton.language as tl

@triton.jit
def _nf4_value(code):
    # All literals are exactly the canonical FP32 table values. Two independent
    # selection chains keep the table visible for the scalar source test.
    negative = tl.where(code == 0, -1.0, -0.6961928009986877)
    negative = tl.where(code == 2, -0.5250730514526367, negative)
    negative = tl.where(code == 3, -0.39491748809814453, negative)
    negative = tl.where(code == 4, -0.28444138169288635, negative)
    negative = tl.where(code == 5, -0.18477343022823334, negative)
    negative = tl.where(code == 6, -0.09105003625154495, negative)
    negative = tl.where(code == 7, 0.0, negative)
    positive = tl.where(code == 8, 0.07958029955625534, 0.16093020141124725)
    positive = tl.where(code == 10, 0.24611230194568634, positive)
    positive = tl.where(code == 11, 0.33791524171829224, positive)
    positive = tl.where(code == 12, 0.44070982933044434, positive)
    positive = tl.where(code == 13, 0.5626170039176941, positive)
    positive = tl.where(code == 14, 0.7229568362236023, positive)
    positive = tl.where(code == 15, 1.0, positive)
    return tl.where(code < 8, negative, positive)
