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

    def _calculate_c(self):
        cumulative_values = np.cumsum(list(self.f.values()))
        return dict(zip(self.f.keys(), cumulative_values))
    
    def _c_inv(self, r):
        pass

    def encode(self, sequence):
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

    def decode(self, state):
        sequence = []
        while state != 0:
            symbol, state = self._decode_step(state)
            sequence.append(symbol)

        return ''.join(sequence)
    
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
    f = {'A': 2, 'B': 1, 'C': 3}
    rans = RangeANSCoder(f)

    message = "ACBCCA"
    compressed_state = rans.encode(message)
    print(compressed_state)

    decompressed_message = rans.decode(compressed_state)
    print(decompressed_message)