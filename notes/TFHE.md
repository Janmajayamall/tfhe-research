# LWE

## Params

An LWE scheme is instantiated with params q,n,p,$\sigma$,$\mu$, where q > p and p | q. q defines the ciphertext modulus, p defines plaintext modulus. For ease both q and p are chosen as power of 2.

## Encoding

A cleartext value $m$ is encoded as plaintext using encoding
$\Delta (m \mod p)$. 

Notice that this results in having $log(q) - log(p)$ bits for noise growth.
## Encryption

First sample secret key $sk \leftarrow \mathbb{B}^n$

Encode message $m \in Z_p$ as plaintext $pt = \Delta m$

Ciphertext $ct \in \mathbb{Z}_q^{n+1}$ for plaintext $pt$ with secret key $sk$ is calculated as
$$ct = (a_0, a_1, ..., a_{n-1}, b) | (a_0, a_1, ..., a_{n-1}) \leftarrow \mathbb{Z}_q, b = \sum_{i=0}^{n-1} a_is_i + e$$

## LWE to TLWE

Discretised TLWE is directly related to LWE when we identify elements in discretised $\mathbb{T}_q \subset \mathbb{T}$ (recall discretised torus as subset of real values in torus with precision $q$) with elements in $\mathbb{Z}_q$. 

Any element $a \in \mathbb{T}_q$ can be written as
$$a = q^{-1} a'$$ where $a' \in \mathbb{Z}_q$.



# GLWE

Note that RLWE and GLWE are same.

## Encryption

Sample secret key $sk = [s_0, ...,s_{k-1}]$ as list of $k$ polynomials $\in \mathbb{Z}_{N, q}$ with binary coefficients. 

Sample $e$ as a polynomial $\in \mathbb{Z}_{N,q}$ with each coefficient sampled from gaussian distribution $N(\sigma^2, \mu)$.

Encrypt plaintext $pt$ as
$$ct = (a_0, a_1, a_2,..., a_{k-1},b) | \space  (a_0, a_1, a_2,..., a_{k-1}) \leftarrow (\mathbb{Z}_q/(X^N + 1))^k, \space b = \sum_{i=0}^{k-1} a_is_i + e + pt$$
$ct \in (\mathbb{Z}_{N,q})^{k+1}$


# GGSW

GGSW is collection of GLWE encryptions. 

## Gadget matrix

Define decomposition base $\beta$ and decomposition level $l$. Note that for accuracy $\beta^l = q$, where $q$ is ciphertext modulus. 
Define gaget matrix $G$ as
$$G^T = \begin{pmatrix}
    \beta^{l-1} & \cdots & \cdots \\
    \vdots & \cdots & \cdots  \\
    \beta^{1} & \cdots & \cdots\\
    \vdots & \cdots & \cdots \\
    & \cdots  & \cdots \\
    & \ddots & \\
     \cdots & \cdots& \beta^{l-1} & \\
     \cdots & \cdots & \vdots   \\
    \cdots & \cdots& \beta^{1} & \\
\end{pmatrix}$$

where $G^T \in \mathbb{Z_q}^{l(k+1) \cdot (k+1)}$

It should be noted that each element in gadget matrix is a constant polynomial. Thus we can write $G^T \in \mathbb{Z_{N,q}[X]}^{l(k+1) \cdot (k+1)}$.
## Encryption

To encrypt message $m \in \mathbb{Z}_{N,p}$, first construct $\pi$ as a column vector with (k+1) rows consisting of GLWE encryptions of 0 under GLWE $sk$.

$$\pi = \begin{pmatrix}
    GLWE_{sk}(0) \\
    GLWE_{sk}(0)\\
	\vdots \\
    GLWE_{sk}(0) \\
\end{pmatrix} = \begin{pmatrix}
    a_{00}, a_{01}, ..., a_{0(k-1)}, b_0 \\
    a_{10}, a_{11}, ..., a_{1(k-1)}, b_1 \\
	\vdots \\
    a_{(lk)0}, a_{(lk)1}, ..., a_{(lk)(k-1)}, b_{lk} \\
\end{pmatrix} \in \mathbb{Z}_{N,q}[X]^{(k+1)l \cdot (k+1)}$$

Then encrypt $m$ as 

$GGSW(m) = \pi + mG^T$


## External product

We can multiply GLWE  encryption of $m_1$ with GGSW encryption of $m_2$ to get GLWE encryption of $m_1m_2$.

Note that GLWE $sk$ must be same for both.

Also note that there exist a function $G^{-1}$ that produces row vector of size $(k+1)l$ which is inverse of $ct_{glwe}$ such that  for a given GLWE ciphertext $G^{-1}(ct_{glwe})G^T = ct_{glwe}$.

More concretely $G^{-1}$ decomposes each of $k+1$ polynomials in $ct_{glwe} = (a_0, .., a_{k-1}, b)$ into $l$ polynomials with decomposition base $\beta$, thus producing a row vector of size $(k+1)l$ as:
$$[a_{00}, a_{01},..., a_{0{l-1}}, a_{10}, ..., a_{1{l-1}}, ..., b_{0}, ..., b_{l-1}]$$

You can think of $G^T$ consisting of recomposition values for each of the decomposed polynomials in $G^{-1}$. For example, recomposition for polynomial $a_0$ is:
$$\sum_{i=0}^{l-1} a_{0i}\beta^{l-{i+1}}$$
where $\beta^{l-{i+1}}$ values are from the first column of $G^T$. 


External product is calculated as: 
$$ct_{ggsw} \cdot ct_{glwe} = G^{-1}(ct_{glwe}) (\pi + m_2G^T) = G^{-1}(ct_{glwe})\pi + G^{-1}(ct_{glwe})mG^T = GLWE(0) + GLWE(m_1m_2)$$

$$ct_{ggsw}(m_2) \cdot ct_{glwe}(m_1) = G^{-1}(ct_{glwe}) (\pi + m_2G^T)$$
$$= G^{-1}(ct_{glwe}(m_1))\pi + m_2G^{-1}(ct_{glwe}(m_2))G^T$$
$$=\pi + m_2ct_{glwe}(m_1)$$
Since $ct_{glwe}(m_1) = \Delta m_1 + e$ and $\pi$ is encryption of 0.
$$ct_{ggsw}(m_2) \cdot ct_{glwe}(m_1) = \Delta m_2m_1 + m_2e$$


TODO: Stop assuming $\beta^l = q$ and rewrite the equations. 

## GLEV and GGSW

Given message $m_1 \in Z_{N, q}$, and decomposition parameters base \beta, and level l, GLEV(m) is: 
$$GLEV(m_1) = [GLWE(m_1\frac{q}{\beta^{1}}),..., GLWE(m_1\frac{q}{\beta^{l}})] \in R_q^{l \times (k+1)}$$

Notice that GLEV encryption is encrypts $m_1$ with different recomposition factors, starting with recomposition factor for most significant digits. 

**GGSW encryption** of $m \in Z_{N,q}$ now becomes a collection of GLEV encryptions as: 
$$GGSW(m) = [GLEV(-S_i\cdot m), ..., GLEV(-S_{k-1} \cdot m), GLEV(m)] \in R_q^{(k+1) \times (l(k+1))}$$
where $S_i$ is $i^{th}$ secret polynomial in GLWE secret key $sk$.

**External product**

Given GLWE(m_1) and GGSW(m_2), we first calculate G^{-1} (GLWE(m_1)) as:
$$G^{-1} (GLWE(m_1)) = G^{-1} [a_0, ..., a_{k-1}, b] = [a_{0,1}, ..., a_{0,l}, a_{1,1}, ..., a_{k-1,l}, b_{1},..., b_{l}]$$
 Then we calculate inner product between $G^{-1} (GLWE(m_1))$ and $GGSW(m)$ as:
 $$\sum_{i=0}^{k-1}<[a_{i,1}, ... a_{i,l}] GLEV(-S_im_2) > + <[b_{1}, ... b_{l}] GLEV(m_2) >$$
Since $[a_{i,1}, ... a_{i,l}]$ are decomposed values of $a_i$ (starting with most significant digit), 
$$<[a_{i,1}, ... a_{i,l}] GLEV(-S_im_2) > = \sum_{j=1}^{l} a_{i,j} GLWE(-Sm_2\frac{q}{\beta^{i+1}})$$
$$= GLWE(-S_ia_im_2)$$
Thus since $b - \sum S_ia_i = \Delta m_1$, 
$$=\sum GLWE(-S_ia_im_2) + GLWE(bm_2) = GLWE(\Delta m_1m_2 )$$

## GLEV approach vs Normal Approach

The only good reason that I am able to come up with for using GLEV approach over normal approach is that GLEV requires $l(k+1)$ multiplication in $Z_{N,q}$ whereas normal approach requires $l(k+1) \times (k+1)$ multiplications (normal approach multiplies row vector with a matrix).

## Ciphertext Multiplication

You can easily multiply a GLWE ciphertext with a scalar by multiplying polynomial with the scalar. However, multiplications between two ciphertexts is not trivial. This is because internal product over Torus is not defined, you need to use external product. 

Let $x \in \mathbb{Z}$ and $a \in \mathbb{T}$. External product between $x$ and $a$ is defined as
$$x \cdot a = a + a + ... + a \space (x \ times)$$

We cannot port this definition directly to encrypted context. In other words, we can cannot add a given GLWE ciphertext by itself $x$ no. of times, where $x$ itself is unknown. To perform ciphertext multiplication between $x \in \mathbb{Z}_{N}[X]$ and $a \in \mathbb{Z}_{N,q}[X]$ we must first encrypt $a$ under GLWE and $x$ under GGSW. Result of their external product equals GLWE encryption of $x \cdot a$, that is 
$$GLWE(x \cdot a) = GGSW(x) \cdot GLWE(a)$$

%%To clear any confusion recall that we view an element defined over discretized torus through its representation in $\mathbb{Z}_q$. That is $a \in \mathbb{Z}_{N,q}[X]$ is defined over $\mathbb{T}_{N,q}[X]$ as 
$$a' = q^{-1}\mathbb{Z}_{N,q}[X]$$%%

## CMUX

TODO

# Bootstrapping

## Sample Extraction GLWE to LWE

Given a GLWE ciphertext $ct_{glwe} \in (\mathbb{Z}_{N,q})^{k+1}$ extracts LWE $ct_{lwe} \in \mathbb{Z}^{kN+1}$ ciphertext of sample at $n^{th}$ index.

Notice that is $ct_{glwe}$ decryted with: 
$$ b - \sum_{i=0}^{k-1} a_is_i$$
where $b, a_i, s_i \in \mathbb{Z_{N,q}}$.

Thus value of sample at index $n$ is equal to
$$m_n =  b_n - \sum_{i=0}^{k-1} \sum_{x=0}^n s_{(i)x}a_{(i)n-x} - (\sum_{j=n+1}^{N-1}s_{(i)j}a_{(i)N+n-j})$$

Let $sk_{lwe} = [s_0[0], ..., s_0[N-1], ..., s_{k-1}[0], ..., s_{k-1}[N-1]]$. 

To construct LWE sample encrypting $m_n$ under $sk_{lwe}$, set $ct_{lwe} = (b', a')$ where, 
$b' = b_n$
$a' = [\sum a_{(i)n},\sum a_{(i)n-1}, ..., \sum a_{(i)0}, \sum -a_{(i)N-1}, -\sum a_{(i)N-2}, ..., \sum -a_{(i)n+1}]$


For implementation refer to glw_sample_extraction of tfhe-rs.

## Test polynomial

Recall that decryption of LWE without removing noise is:
$$\mu = b - \sum{a_is_i} \mod{q}$$ 

The intuition behind blind rotation is to construct a polynomial as: 
$$X^{-b + \sum{a_is_i}}$$
In other words, we obtain $-\mu$ in the exponent of the polynomial. We the multiply the obtained polynomial with $v(x)$:
$$X^{-\mu} \cdot v(x)$$
to get $\mu^{th}$ coefficient of v(x) as constant term (ie rotate v left by $\mu$ positions).

But notice that
1. $X \in Z_{N,q}[X]$, thus $X$ is of order $2N$ in $Z_{N,q}[X]$ (ie $X^{2N}=1$). This implies instead of $\mu$, $-\hat\mu$ must be calculated as: 
   $$-\hat{b} + \sum{\hat{a_i}s_i} \mod 2N$$which is mod 2N instead of mod q, where $\hat{b}$ and $\hat{a_i}$ are approximate values in mod 2N calculated as
   $$\hat{b} = round(\frac{2N \cdot b}{q}) \mod 2N$$
   and 
   $$\hat{a_i} = round(\frac{2N \cdot a_i}{q}) \mod 2N$$
   Note that error induced by approximation is called *drift*. It can be reduced by choosing appropriate parameters.
2. Another problem is that since $v(x)$ can have only $N$ coefficients we can only map $N$ encode $N$ values in $v(x)$. This requires us to limit magnitude of $-\hat\mu$ to $N$ values. This can be achieved limiting $\mu = b + \sum a_is_i$ (decryption with noise) to $q/2$. In practice we this translates to setting carry bit greater than or equal to 1, so that noise less encoded plaintext can have a maximum value $q/2$.
3. Another thing to notice is that that $-\hat\mu$ consists of noise (it is decryption without rounding). Recall that in decryption procedure, a range values will map to same message and to obtain the message from decrypted plaintext we need to round it (Imagine the torus). Similarly, when encoding coefficients in $v(x)$ we encode the same plaintext over a range of coefficients. This ensures that rotation by $-\hat\mu$ (which contains errors) maps to correct encoded plaintext coefficient. 
4. Following from (3), recall that $\mu$, thus $\hat\mu$,can be negative as well. This becomes a problem for values at the border (ie values corresponding to 0). Imagining the torus again and recall that negative values upto error range beyond 0, must round back to 0. We need to assure that the same is handled in $v(x)$. If  $\hat\mu$ is negative it will rotate $v(x)$ by right. To make sure that value corresponding to 0 land in coefficient of $v(x)$ we need to encode some of higher coefficients with value corresponding to 0. Moreover, since rotation towards right wraps higher coefficients around, thus negating them ($v(x)$ is negacylic) we need to negate the values encoded in higher coefficients. This assures that upon wrapping around, we get correct positive values in $v(x)$'s constant term. 
   

In practice test polynomial is constructed as (we view polynomial with coefficient vector): 
1. Construct vector $m$ consisting of all possible values in message space $[0, 2^M)$, as:
   $$[0,..,2^M-1]$$
2. Divide N into $b$ blocks where $b = N/(2^M)$ and fill each block with repeated values of values in $m$ (this corresponds to mapping same noiseless values over the error range). 
   $$[0,0,0,0,1,1,1,1,...,2^M-1]$$ where $b = 4$.
3. Slice half of block corresponding to 0, negate its values and append it at the end. 
4. Encode the resulting coefficient as a plaintext, since we view them as a valid GLWE ciphertext. This means multiply each value by $\Delta$.

Note that replacing vector $m$ with an output of a function, turns bootstrapping into programmable bootstrapping (PBS). That is for function $f(x) | x \in [0, 2^M)$, construct $m$ as
$$m = [f(0), f(1), ..., f(2^M-1)]$$

It is worth noting that boostrapping is PBS with identity function. 

## Blind rotation

Blind rotation simply equates to calculating $X^{-\mu} v(x)$ homomorphically. 

Once we have constructed test polynomial $v(x)$ correctly, we can repeatedly use CMUX operations to perform blind rotations. Recall that CMUX operation is as: 

Given GGSW encryption of bit $b$ and two GLWE encryption, $a$ and $c$, CMUX operation outputs:
$$o = CMUX(b, a, c)$$
if $o == a$ if $b == 1$, c otherwise. 

We start with trivial GLWE encryption of $v(x)$. This means we construct GLWE encryption of $v(x)$ as:
$$ggsw_{sk} = GLWE(v(x)) = (0, 0,..., 0, v(x)) \in Z_{N,q}[X]^{k+1}$$
We will additionally need GGSW encryptions of each bit of LWE secret key $s_{lwe}$, that is
$$[GGSW(s_0), ..., GGSW(s_n)] \space for \space s_{lwe} = [s_0, ..., s_n]$$
We perform blind rotation as: 
```latex
let acc = X^{-\hat{b}}GLWE(v(x))
for i in 0..n { 
	acc = CMUX(ggsw_sk, X^{\hat{a_i}}acc, acc)
}
```

Notice that we start with $X^{-\hat{b}}v(x)$ and $\hat{a_i}$ is only added to $-\hat{b}$ in the exponent if $s_i$ is 1, otherwise not. Thus we end up with GLWE encryption of $X^{-\hat\mu}v(x)$

Notice that GLWE ciphertext $X^{-\hat\mu}v(x)$ encrypts desired value $\mu$ which corresponds to correct value in message space on the constant term. To extract the constant term as LWE ciphertext from GLWE ciphertext we use sample extraction with $0^th$ term.

## Key Switching

Key Switching switches and LWE ciphertext encrypted under $s'$ to encryption under $s$.

Given an LWE ciphertext $c = (a_0, ..., a_{n-1}, b)$ encrypted under $s'$ recall that decryption procedure as: 
$$\Delta(m) + e = b - \sum_{i=0}^{n-1}a_is'_i$$
The idea is to calculate the summation homomorphically using encryption of each bit of $s'$ under $s$. However, doing this naively will blow up the resulting noise since we multiply LWE encryption $LWE(s'_i)$ with a scalar in ciphertext space (ie $a_i \in Z_q$). To perform key switching while keeping noise growth under control we use digit decomposition. 

### Digit decomposition

Digit decomposition is implement using gadget decomposition matrix. Given a base $\beta$ and levels $l$ we construct the following gadget matrix: 
$$G = [\beta^{l-1},...,\beta^{0}]$$
Given base $\beta$ decomposition of $a \in Z_q$ as: 
$$G^{-1}(a) = [a_0,...,a_{l-1}]$$
where $a_i$ are extracted starting from most significant bits. For ex, $a_0$ corresponds to most significant $log(\beta)$ bits of $a$.

We can reconstruct $a$ upto $l$ level accuracy as
$$a = G^{-1}(a)G^{T}$$
Note that in practice we choose $\beta * l$ smaller than $q$ and safely ignore LSBs since they consist of noise. 

### Key Switching Key

To perform key switching we require LWE encryption each bit of $s'$. To use digit decomposition and recomposition during key switching we decompose each bit in $s'$ and encrypt them. That is, 
1. Decompose $s'_i$ using digit decomposition. Since $s_i$ is a bit, in practice this is omitted. 
2. Given gadget row vector $G$, calculate $s'_iG = [s'_i\beta^{l-1}, ..., s'_i]$.
3. Encrypt each element of $s'_iG$ using $s$. This produces $l$ LWE ciphertexts. 
After encrypt all bits in $s'$ we obtain an 2D array of LWE ciphertext where $j^{th}$ column of $i^{th}$ row is an LWE encryption as $LWE_s(s'_i\beta^{l-{j+1}})$. Let's call this 2D array key switching key $ksk$.

We then perform key switching for LWE encryption $LWE_{s'}(m) = (a'_0,...,b')$ as
$$LWE_s(m) = (0,...0, b') - \sum_{i=0}^{n-1} G^{-1}(a_i')ksk[i]^T$$
Notice that $G^{-1}(a_i')ksk[i]^T$ expands to 
$$[a'_{i0},...,a_{i(n-1)}'][LWE_{s}(s'_i\beta^{l-1}),...,LWE_{s}(s'_i)]^T = LWE(a'_is_i)$$
but without accumulating large noise since magnitude of each scalar value $a'_{il}$ is limited to base $\beta$.

We use key switching in bootstrapping to switch LWE ciphertext extracted in sample extraction from encryption under $s' \in \mathbb{B}^{kN}$ to encryption under $s \in \mathbb{B}^n$. This means ksk will be a 2D matrix as $ksk \in (\mathbb{Z}^{n+1}_q)^{(kN \cdot l)}$ 


## Bootstrapping Step

Assuming secure parameters for $LWE = (n, p, q)$ and $GLWE = (k, N, p , q)$ and decomposition parameters base $\beta$ and level $l$. We will need the following things for bootstrapping:
1. LWE ciphertext encrypting message $m \in Z_p$ under sk $s$ as $LWE_s(m) = (a_0, ..., a_{n-1}, b)$. This is the ciphertext we will bootstrap. 
2. Bootstrapping key which consists of keys for blind rotation and key switching key.
	1. Keys for blind rotation: Recall that blind rotation requires GGSW encryption of each bit of LWE secret key $s$ under GLWE secret key $s' \in \mathbb{Z}_{N,q}^k$. Thus we require $n$ GGSW ciphertexts as:
	   $$[GGSW_{s'}(s_0), ... GGSW_{s'}(s_i)]$$
	2. Key switching key for switching LWE ciphertext $LWE_{s'}(m)$ to $LWE_s(m)$. 
3. Test polynomial $v(x)$.

Bootstrapping procedure proceeds as
1. Blind rotate $LWE_s(m)$ using test vector polynomial $v(x)$. This results in GLWE ciphertext encrypted under $s'$. Note that constant term of the ciphertext equates to LWE ciphertext of the desired message encrypted under $s'$.
2. Sample extract $0^{th}$ term from GLWE ciphertext obtained in (1) to get an LWE ciphertext of $0^th$ term encrypted under $s'$. Note that extracted LWE ciphertext is encryption of desired message with reduced noise under sk $s'$.
3. Key switching LWE ciphertext in (2) from encryption under $s'$ to encryption under $s$.

After (3) we have LWE encryption of desired message with reduced noise as $LWE_s(m)$. 

Note that we view GLWE secret key $s' \in Z_{N,q}^k$ as an LWE secret key $s' \in Z_q^{kN}$ by interpreting coefficients as vectors. 

Programmable bootstrapping follows the same procedure by filling in $m$ vector when constructing test polynomial $v(x)$ with values of $f(x)$ for $x \in [0, p)$ instead. 


# Decomposition

## Gadget vector

For ciphertext space $q$, base $\beta$ and level $l$, we represent gadget vector as:
$$g = [\frac{q}{\beta}, \frac{q}{\beta^2} ..., \frac{q}{\beta^{l}}]$$
On a side note, for clarity when $\beta * l = q$ we can write g as:
$$g = [\beta^{l-1},..., 1]$$
Given an unsigned integer $a \in Z_q$ we can write its decomposition vector of size $l$ as: 
$$g^{-1}(a) = [a_0, a_1, ..., a_{l-1}] \in Z_{\beta}^l$$
where we start decomposition with MSBs. This implies $a_0$ is $log(\beta)$ MSBs of $a$.

To recompose $a$, we can calculate the following: 
$$g^{-1}(a)g \approx \sum_{i=0}^{l-1} a_i(\frac{q}{\beta^i})$$

Note that we $\beta * l = q$, then recomposition will be exact.

## Why Decomposition

Decomposition is one the techniques used to decrease noise growth when multiplying a ciphertext with a plaintext. For example, let $ct = LWE_s(m) = (a_0,...,b)$. We can multiply $c$ by $ct$ to produce $LWE_s(cm)$, that is a ciphertext encrypting product $cm$. However notice that after multiplication noise in the ciphertext grows since: 
$$c(ct) = c(a_0, ..., b) = (ca_0, ..., cb)$$
and decrypting c(ct) gives the following output: 
$$c(b - \sum a_is_i) = c(m + e) = cm + ce$$
Thus the noise in ciphertext after multiplication grows by magnitude of $c$.

We use decomposition to reduce the noise growth. 

Instead of LWE encryption of $m$, we will first multiply $m$ with gadget vector $g$ to produce $mg$ as 
$$mg = [m\frac{q}{\beta}, m\frac{q}{\beta^2} ..., m\frac{q}{\beta^{l}}]$$
Then we will encrypt each element of $mg$ to produce $l$ LWE ciphertexts 
$$g_{lwe} = [LWE_s(m\frac{q}{\beta}), LWE_s(m\frac{q}{\beta^2}) ..., LWE_s(m\frac{q}{\beta^{l}})]$$
To obtain product $cm$ for $c \in Z_q$, we calculate the following: 
$$g^{-1}(c)g_{lwe}^T = \sum c_i LWE_s(m\frac{q}{\beta^{i+1}})$$
Since $c_i LWE_s(m\frac{q}{\beta^{i+1}}) = LWE_s(c_i m\frac{q}{\beta^{i+1}})$, summation of all products results into $LWE_s(m(\sum c_i\frac{q}{\beta^{i+1}})) = LWE_s(mc)$.

Note that this time noise growth is $\sum c_i e_i$ which depends on maximum value of $c_i \in Z_{\beta}$.

Decomposition trick to reduce noise growth can also be applied to GLWE setting. 


## Unsigned decomposition

Unsigned decomposition is what you will normally think when decomposing an unsigned integer. For ex, decomposition of $a \in Z_q$ is as:
$$g^{-1}(a) = [a_0, a_1, ..., a_{l-1}] \in Z_{\beta}^l$$
starting with MSBs of $a$.
## Signed Decomposition

Signed decomposition further improves upon unsigned decomposition in terms of noise growth. Instead of each element of decomposed vector being $\in [0, \beta)$ as in unsigned decomposition, each element is $\in [-\beta/2, \beta/2)$. 

To construct signed decomposition of $a \in Z_q$ the idea is to convert decomposed values to their signed representation, but we need to take of carry overs. For example, let decomposition of $a$, starting from LSB, be $[a_0, a_1, ..., a_{l-1}]$. To obtain signed decomposition such that each value $a_i$ is converted to its signed representation $\in [-\beta/2, \beta/2)$ we subtract $\beta/2$ from $a_i$ if $a_i \geq \beta/2$ . If the $a_{i}$ is subtracted then we must add (ie carry over) 1 to $a_{i+1}$ and then convert $a_{i+1}$ to its signed representation. This works because: 
$$a_{i-1} \beta^{i-1} + a_{i} \beta^{i} = a_{i-1} \beta^{i-1} - \beta^i + a_{i} \beta^{i} + \beta^i = \beta^{i-1}(a_{i-1} - \beta) + \beta^{i}(a_i + 1)$$

There are two effects of using signed decomposition: 
1. each value $a_i$ in its signed representation is in $\in [-\beta/2, \beta/2)$ and has maximum possible magnitude $\beta/2 - 1$. This means noise growth is half of what will be using unsigned decomposition. 
2. The maximum representable value, ie $a$, is reduced to $\sum (\beta/2 - 1)\beta^i$. This means we need to limit the value of $a$. (I believe) in practice this is achieved by having $\geq 1$   carry bits. Check [this](https://jeremykun.com/2021/12/11/the-gadget-decomposition-in-fhe/) post for more information.







TFHE-rs TODO
1. Under polynomial karatsuba multiplication [here](https://github.com/zama-ai/tfhe-rs/blob/80b5ce7f63a985be2a7d3bc636729e1c8e24a334/tfhe/src/core_crypto/algorithms/polynomial_algorithms.rs#L549).
2. Replace definitions of base decompositions such that we don't have to assume $\beta l = q$.
3. 

