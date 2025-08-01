from typing import Any, Optional
from collections import Counter

import numpy as np

Bitstream = list[int]

class RangeANSCoder:
    def __init__(
            self,
            f: dict[Any, int],
            k: Optional[int] = 12,
            l: int = 16,
            flush_size: int = 1,
            range_factor: int = 16,
    ):
        """
        Initialize the coder, notably with alphabet frequency.

        Args:
            f: dict[Any, int]
                A dictionary mapping each symbol to their integer frequency
                counts. Does not have to be in any particular order, as
                ordering is optimized by the object.
            k: int
                Value used to scale F normalization. F will be normalized so
                that sum(F) = M = 2^k' for the nearest 2^k' superior to M and
                inferior to 2**k. Normalizing F in this way speeds up decoding,
                specifically when dividing by M, at the small cost of
                compression efficiency (we are approximating the probability
                mass distribution, and potentially increasing the value of M).
                For these reasons, it may be wiser to not normalize F for short
                messages, where decoding time is less important and increasing
                the value of M would decrease efficiency too greatly.
                Default is 12. None does not normalize F.
            l: int
                Used as power of 2 to calculate lower bound for state size.
                If X < L during streaming, the state will be renormalized. Will
                be used to define H = L x M.
            flush_size: int
                Number of bits to flush when the encoded state becomes superior 
                to H.

        """
        self.f = dict(sorted(f.items(), key=lambda x: -x[1])) # highest frequency symbols first
        self.a = list(self.f.keys())
        self.m = sum(list(f.values()))

        if k is not None:
            self._normalize_f(k)

        self.c = self._calculate_c()
        self.reverse_c = self._reverse_c()
        
        self.flush_size = flush_size
        self.flush_mask = (1 << self.flush_size) - 1

        self.range_factor = 1 << range_factor
        self.l: int = self.range_factor * self.m
        self.h: int = self.l * (1 << self.flush_size)

        self.max_shrunk_state = {}
        for s in self.a:
            freq = self.f[s]
            self.max_shrunk_state[s] = self.range_factor * freq * (1 << self.flush_size) - 1

        self.NUM_STATE_BITS = self.h.bit_length()
        self.DATA_BLOCK_SIZE_BITS = 32

    def _get_highest_k(self, k):
        for candidate_k in range(k):
            if 2**candidate_k > self.m:
                return candidate_k
        return k

    def _normalize_f(self, k):
        """
        We normalize F so that sum(F) = M = 2^k.
        """
        k = self._get_highest_k(k)
        new_m = 2**k

        # Rounding may not always bring the sum 2**k, not a very robust implementation
        self.f = {k: round((v / self.m)*(new_m)) for k, v in self.f.items()}
        self.m = sum(self.f.values())


    def _calculate_c(self) -> dict[Any, int]:
        """
        We calculate a list of intervals (practically, the cumulative sum)
        given the symbol frequencies.

        Example:
            Given F = {A: 2, B: 1, C: 3}
            We find C = {A: 0, B: 2, C: 3}
        """
        # we need to add 0 and remove the last value, so our C gives the start
        # of each interval and not the end
        values = [0, *list(self.f.values())[:-1]]
        cumulative_values = np.cumsum(values)
        return dict(zip(self.f.keys(), cumulative_values))
    
    def _reverse_c(self):
        """
        We reverse the list of intervals to use as a lookup table in the
        inverse CDF function.
        """
        reverse_c = dict(sorted(self.c.items(), key=lambda x: -x[1]))
        return reverse_c
    
    def _c_inv(self, r: int) -> Any:
        """
        Inverse CDF function used to find symbols during decoding. We can think
        of this as finding the interval in which an integer falls, to find the
        character associated with it.

        Example:
            Given C = {A: 0, B:2, C: 3}
            CDF_inv(1) = A
            CDF_inv(2) = B
            CDF_inv(3) = C
        """
        for key in self.reverse_c.keys():
            if self.reverse_c[key] <= r:
                return key
            
    def _encode_symbol(self, state: int, symbol: Any) -> int:
        F_s = self.f[symbol]
        C_s = self.c[symbol]

        d = state // F_s
        r = state % F_s
        state = d * self.m + C_s + r
        
        return state
    
    def _decode_symbol(self, state: int) -> tuple[Any, int]:
        # first we decode the symbol
        r = state % self.m
        symbol = self._c_inv(r)
        
        # then we find the next state
        F_s = self.f[symbol]
        C_s = self.c[symbol]
        d = state // self.m
        next_state = d * F_s + r - C_s

        return symbol, next_state
    
    def _shrink_state(
            self,
            state: int,
            next_symbol: Any,
    ) -> tuple[int, Bitstream]:
        """Shrink state by streaming out lower bits."""
        out_bits = []

        while state > self.max_shrunk_state[next_symbol]:
            out_bits.insert(0, state & ((1 << self.flush_size) - 1))
            state >>= self.flush_size

        return state, out_bits

    def _expand_state(
            self,
            state: int,
            bitstream: Bitstream,
            bits_consumed: int
    ) -> tuple[int, int]:
        """Expand state by reading bits from the stream."""
        num_bits_read = 0
        while state < self.l:
            bits_to_read = bitstream[bits_consumed + num_bits_read: bits_consumed + num_bits_read + self.flush_size]
            val = 0
            for i, bit in enumerate(bits_to_read):
                val |= bit << (len(bits_to_read) - 1 - i)

            state = (state << self.flush_size) | val
            num_bits_read += self.flush_size
        return state, num_bits_read

    def encode(self, sequence: list) -> int:
        """
        Encode a sequence of symbols into a single integer state.
        """
        state = 0
        for symbol in sequence:
            state = self._encode_symbol(state, symbol)
        return state

    
    def decode(self, state: int, n_symbols: int) -> list:
        sequence = []
        for _ in range(n_symbols):
            symbol, state = self._decode_symbol(state)
            sequence.append(symbol)

        # sequence is decoded in reverse 
        reverse_sequence = sequence[::-1]
        return reverse_sequence

    def stream_encode(
            self,
            sequence: list,
        ) -> Bitstream:
        """
        Encode a sequence of symbols into a byte array that can be written to a
        file. Since this a stream encoding, we can accept an input sequence of
        arbitrary length.
        """ 
        bitstream = []
        state = self.l

        for symbol in sequence:
            state, out_bits = self._shrink_state(state, symbol)
            bitstream = out_bits + bitstream

            state = self._encode_symbol(state, symbol)
        
        final_state_bits = list(map(int, bin(state)[2:].zfill(self.NUM_STATE_BITS)))
        bitstream = final_state_bits + bitstream

        size_bits = list(map(int, bin(len(sequence))[2:].zfill(self.DATA_BLOCK_SIZE_BITS)))
        bitstream = size_bits + bitstream

        return bitstream
        

    def stream_decode(
            self,
            bitstream: list,
    ) -> list:
        """Decode a stream of data."""
        size_bits = bitstream[:self.DATA_BLOCK_SIZE_BITS]
        num_symbols = int("".join(map(str, size_bits)), 2)
        bits_consumed = self.DATA_BLOCK_SIZE_BITS

        state_bits = bitstream[bits_consumed : bits_consumed + self.NUM_STATE_BITS]
        state = int("".join(map(str, state_bits)), 2)
        bits_consumed += self.NUM_STATE_BITS

        decoded_sequence = []
        for _ in range(num_symbols):
            symbol, state = self._decode_symbol(state)

            state, num_bits_read = self._expand_state(state, bitstream, bits_consumed)
            bits_consumed += num_bits_read

            decoded_sequence.insert(0, symbol)

        assert state == self.l, f"Final state {state} does not match initial state {self.l}."
        return decoded_sequence

if __name__ == '__main__':

    # f = {'A': 2, 'B': 1, 'C': 3}
    # message = ["A", "C", "B", "C", "C", "A"] * 4
    original_message = "George Washing Machine"
    message = list(original_message) * 1000
    f = Counter(message)

    print(f"Message: {original_message}")

    rans = RangeANSCoder(f)

    # --------------------------------
    # ----- NORMAL ENCODE/DECODE -----
    # --------------------------------

    # compressed_state = rans.encode(message)
    # print(f"Compressed state: {compressed_state}")

    # decompressed_message = rans.decode(compressed_state, len(message))
    # print(f"Decoded message: {decompressed_message}")

    # --------------------------------
    # ----- STREAM ENCODE/DECODE -----
    # --------------------------------

    compressed_stream = rans.stream_encode(message)
    print(f"Compressed bitstream: {[int(x) for x in compressed_stream]}")
    
    print("\n")
    print(f"Original message length in bytes: {len(message)}\nCompressed bitstream length in bytes: {len(compressed_stream) / 8}")
    print("\n")

    decompressed_message = rans.stream_decode(compressed_stream)
    print(f"Decompressed message from stream: {"".join(decompressed_message)}")
    print(f"Equality check: {message == decompressed_message}")