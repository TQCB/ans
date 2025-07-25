from typing import Iterable

import numpy as np

class RangeANSCoder:
    def __init__(
            self,
            f: dict[str, int],
            k=None,
    ):
        self.f = dict(sorted(f.items(), key=lambda x: -x[1])) # highest frequency symbols first
        self.k = k

        self.m = np.sum(list(f.values()))
        self.c = self._calculate_c()
        self.reverse_c = self._reverse_c()

    def _calculate_c(self):
        # we need to add 0 and remove the last value, so our C gives the start
        # of each interval and not the end
        values = [0, *list(self.f.values())[:-1]]
        cumulative_values = np.cumsum(values)
        return dict(zip(self.f.keys(), cumulative_values))
    
    def _reverse_c(self):
        reverse_c = dict(sorted(self.c.items(), key=lambda x: -x[1]))
        return reverse_c
    
    def _c_inv(self, r) -> str:
        for key in self.reverse_c.keys():
            if self.reverse_c[key] <= r:
                return key

    def encode(self, sequence: Iterable):
        state = 0
        for symbol in sequence:
            state = self._encode_step(state, symbol)
        return state

    def _encode_step(self, previous_state, symbol):
        F_s = self.f[symbol]
        C_s = self.c[symbol]

        d = np.floor(previous_state / F_s)
        r = np.mod(previous_state, F_s)
        state = d * self.m + C_s + r

        return state

    def decode(self, state, n_symbols):
        sequence = []
        for _ in range(n_symbols):
            symbol, state = self._decode_step(state)
            sequence.append(symbol)

        # sequence is decoded in reverse 
        reverse_sequence = sequence[::-1]
        return reverse_sequence
    
    def _decode_step(self, state):
        # first we decode the symbol
        r = np.mod(state, self.m)
        symbol = self._c_inv(r)
        
        # then we find the next state
        F_s = self.f[symbol]
        C_s = self.c[symbol]
        d = np.floor(state / self.m)
        next_state = d * F_s + r - C_s

        return symbol, next_state

if __name__ == '__main__':
    message = ["A", "C", "B", "C", "C", "A"]
    f = {'A': 2, 'B': 1, 'C': 3}

    rans = RangeANSCoder(f)

    print(f"Message: {message}")

    compressed_state = rans.encode(message)
    print(f"Compressed state: {compressed_state}")

    decompressed_message = rans.decode(compressed_state, len(message))
    print(f"Decoded message: {decompressed_message}")