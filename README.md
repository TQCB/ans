# Asymmetric Numeral Systems (ANS)

## Range ANS (rANS)

### Introduction
Consider a simple example where we want to reprensent a sequence of digits $S = \{ 5, 2, 9, 0, 8, 1 \}$, where each digit is in the range $[0, ..., 9]$, in a single interger state. To do so, we could use a single number $X_6 = 529081$, which is just the base-10 composition of our digits:
$$X_0 = 0$$
$$X_1 = X_0 \times 10 + 5$$
$$X_2 = X_1 \times 10 + 2$$
$$...$$
$$X_6 = X_5 \times 10 + 1$$

Our final state $X_6$ contains all information necessary to recover our original sequence, and can be encoded in binary using $log_2(X_6)$ bits. It can be decoded in reverse order recursively:

$$X_6 = 529081$$
$$X_5 = \lfloor X_6/10 \rfloor, s_6 = mod(X_6, 10)$$
$$X_4 = \lfloor X_5/10 \rfloor, s_5 = mod(X_5, 10)$$
$$...$$
$$X_0 = 0, s_1 = 5$$

This system works well but fails when we want to optimally encode sequences with a non-uniform symbol distribution. At time step $t$, we multiply the size of the state by roughly 10 since $X_t \approx X_{t-1} \times 10$. This is optimal if all symbols $[0, ..., 9]$ are equiprobable, as we use $log_2 10$ bits per encoded symbol. In the case of a non-uniform symbol distribution, our entropy can be far lower than this.

To generalize our structure to cases with non-uniform distributions, we can scale our most frequent symbols with a factor smaller than 10 and the least frequent by a larget factor. This is how rANS allows us to achieve efficient representations of a sequence in a single integer state.

### Notation

1. $S = (s_1, s_2, s_3, ..., s_n)$ are the input string of symbols from alphabet $A = \{a_1, a_2, ..., a_k\}$
2. $F = \{F_{a_1}, F_{a_2}, ..., F_{a_k}\}$ are the integer frequency counts proportional to the probability mass distribution $\{p_1, p_2, ..., p_k\}$ of our symbols. Additionally $M = \Sigma^{k}_{i=1}F_i$ , and thus $p_i = \frac{F_{a_i}}{M}$
3. $C_{a_i} = \Sigma^{i-1}_{j=1}F_{a_j}$ corresponds to the cumulative distribution of the symbols

### Explanation

One can generalize this process in 2 steps:

First,
- We choose an $M$ sized block.
- For state $X_{t-1}$ we choose a $block$ $\lfloor \frac{X_{t-1}}{F_{s_t}} \rfloor$.

Secondly,
- We choose a $section$ out of the $M$ allowed integers from the block.
- For a symbol $s_t$, the $section$ will be in the range $[C_{s_t}, C_{s_{t+1}} - 1]$

The next state $X_t$ is composed as $X_t = block \times M + section$

Since we have a distinct range for every symbol, we can decode $s_t$ using only $section = mod(X_t, M)$ and a function that returns a symbol based on which $C$ range it falls into. We can also find the $block$ from $X_t$ using $block = \lfloor \frac{X_t}{M} \rfloor$. Since $block = \lfloor \frac{X_{t-1}}{F_{s_t}} \rfloor$, all we need to find $X_{t-1}$ is $mod(X_{t-1}, F_{s_t})$. To do this, we choose our $section$ as $C_{s_t} + mod(X_{t-1}, F_{s_t})$ which is in the range $[C_{s_t}, C_{s_{t+1}}-1]$ when encoding.

### Encoding
The integer state $X$ of our sequence after seeing $t$ symbols can be defined using the encoding step $C$, using the previous state and the current symbol:

$$X_t = C(X_{t-1}, s_t)$$

This encoding step is defined as:

$$C(X_{t-1}, s_t) = \lfloor \dfrac{X_{t-1}}{F_s} \rfloor \times M + mod(X_{t-1}, F_s) + C_s$$

### Decoding

As seen earlier, we can recursively decode our final compressed state in reverse. To do so, we use a decoding step:

$$r = mod(X_t, M)$$
$$s_t = C_{inv}(r)$$
$$X_{t-1} = \lfloor \dfrac{X_t}{M} \rfloor \times F_{s_t} + r - C_{s_t}$$

where $C_{inv}$ is the inverse function of cumulative frequency, so that $C_{inv}(y) = a_i, if C_{a_i} \le y < C_{a_i+1}$.