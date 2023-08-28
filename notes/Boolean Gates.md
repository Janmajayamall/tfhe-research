
Recall that programmable bootstrapping (PBS) can evaluate an arbitrary univariate function over tfhe ciphertext while reducing the noise. This means we can evaluate boolean gates on ciphertext using PBS. Before PBS we first require to convert multivariate function evaluation to univariate function. We achieve this using simple scalar multiplication. For ex, consider two ciphertext $ct_0$ and $ct_1$ and we want to evaluate AND gate over them such that output is ecnryption of $m_0 \wedge m_1$. We must restrict the plaintext space to 2 bits with padding bits set to 1. Then we proceed as follows: 
1. Generate test vector $v(x)$ that maps all possible inputs in plaintext space to their respective outputs after evaluating AND gate. MSB is considered left input to the gate and LSB is considered right input. 
2. Given $c_0$ and $c_1$, calculate $c_{in} = 2 * c_1 + c_0$, where $c_1$ encrypts left input and $c_0$ encrypts right input. Note that $c_{in}$ has left input in MSB and right inputs in LSB. 
3. Evaluate PBS over $c_{in}$ with test vector $v(x)$. Output of PBS encrypts $m_0 \wedge m_1$. 

We can follow the same procedure for any other 2 input gate. 

We can extend to 3 input gate by increasing plaintext space to 3 bits and mapping bits from MSB to LSB as left to right input to 3 input gates. Given three ciphertexts you must now calculate $c_{in}$ as $c_{in} = 4 * c_2 + 2 * c_1 + c_0$, where $c_2, c_1, c_0$ encrypt bits from leftmost to rightmost input to the gate. 

It is easy to observe that, given accurate parameters, we can extend to gates with higher no. inputs by performing PBS over c_{in} calculated as binary recomposition of ciphertexts encrypting bits from MSB to LSB ans MSB map to leftmost gate and LSB maps to rightmost gate. 

