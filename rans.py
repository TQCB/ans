from typing import Any, Optional

import numpy as np

class RangeANSCoder:
    def __init__(
            self,
            f: dict[Any, int],
            k: Optional[int] = 12,
            l: int = 16,
            flush_size: int = 1,
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
        self.m = sum(list(f.values()))
        
        self.l: int = 2**l
        self.flush_size = flush_size
        
        if k is not None:
            self._normalize_f(k)

        self.c = self._calculate_c()
        self.reverse_c = self._reverse_c()

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
        self.f = {k: round((v / self.m)*(new_m)) for k, v in self.f.items()}
        self.m = new_m


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

    def encode(self, sequence: list) -> int:
        """
        Encode a sequence of symbols into a single integer state.
        """
        state = 0
        for symbol in sequence:
            state = self._encode_symbol(state, symbol)
        return state

    def _encode_symbol(self, state: int, symbol: Any) -> int:
        F_s = self.f[symbol]
        C_s = self.c[symbol]

        d = state // F_s
        r = state % F_s
        state = d * self.m + C_s + r
        
        return state

    def decode(self, state: int, n_symbols: int) -> list:
        sequence = []
        for _ in range(n_symbols):
            symbol, state = self._decode_symbol(state)
            sequence.append(symbol)

        # sequence is decoded in reverse 
        reverse_sequence = sequence[::-1]
        return reverse_sequence
    
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

    def stream_encode(
            self,
            sequence: list,
        ) -> tuple[int, list]:
        """
        Encode a sequence of symbols into a byte array that can be written to a
        file. Since this a stream encoding, we can accept an input sequence of
        arbitrary length.
        """
        
        stream = []
        state: int = self.l # we initialize to L to make sure we are always within L < X_t < H

        for symbol in sequence:
            while state >= self.l * self.f[symbol]:
                bits_to_write = state & ((1 << self.flush_size) - 1) # bitmask last flush_size bits of state
                stream.append(bits_to_write)

                print(f"state before shift: {state}")
                state >>= self.flush_size
                print(f"state after shift {state}")

            state = self._encode_symbol(state, symbol)
            print(f"Encoding iteration {symbol}: State = {state} Bitstream = {stream}")

        return state, stream # return final state and bitstream
    
    def stream_encode_to_file(
            self,
            sequence: list,
            file_path,
    ) -> None:
        return
        with open(file_path, 'wb'):
            pass


    def stream_decode(
            self,
            state: int,
            bitstream: list,
            n_symbols: int,
    ) -> list:
        print("================")
        print("START OF DECODING")
        print(f"{state=}\n{bitstream=}\n{n_symbols=}")
        print("================")

        stream_idx = len(bitstream) - 1
        sequence = []

        for i in range(n_symbols):
            symbol, state = self._decode_symbol(state)
            sequence.append(symbol)
            print(f"Iteration: {i} Decoded symbol: {symbol} New state: {state}")

            # check that state isn't inferior to L and still have bits left
            while state < self.l * self.m and stream_idx >= 0:
                if stream_idx < 0:
                    raise ValueError(f"Ran out of bits while decoding symbol {i}")
                print("Reading from bitstream, old state: {state}", end=" ")
                bits = bitstream.pop()
                state = (state << self.flush_size) | bits
                print(f"New state: {state}, read bits: {bits}")
                stream_idx -= 1

        # sequence is decoded in reverse 
        reverse_sequence = sequence[::-1]
        return reverse_sequence

if __name__ == '__main__':

    f = {'A': 2, 'B': 1, 'C': 3}
    message = ["A", "C", "B", "C", "C", "A"] * 4
    # message = ["A", "B", "C"]
    print(f"Message: {message}")

    rans = RangeANSCoder(f)

    print(rans.f, rans.c)

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

    final_state, compressed_stream = rans.stream_encode(message)
    print(f"Final state: {final_state}\nCompressed bitstream: {compressed_stream}")

    decompressed_message = rans.stream_decode(final_state, compressed_stream, len(message))
    print(f"Decompressed message from stream: {decompressed_message}")
    print(f"Equality check: {message == decompressed_message}")