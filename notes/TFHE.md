# LWE

## Params

An LWE scheme is instantiated with params $q,n,p,\sigma_l$ where $q >> p$ and $p \vert q$. 

$q$ is the ciphertext modulus, $p$ is the plaintext modulus. For ease both $q$ and $p$ are chosen as power of 2.

Note: most CPU based implementations set $q$ to $2^{64}$ or $2^{32}$. This is convenient because word-size of modern cpus are either 32 bits or 64 bits. 

## Encoding

A cleartext value $m$ is encoded as plaintext with encoding

$\Delta (m \mod p)$ 

where $\Delta = q/p$

Notice that this roughly gives $log(q) - log(p)$ bits for noise growth.

## Encryption

First sample secret key $sk \leftarrow \mathbb{B}^n$

Encode message $m \in Z_p$ as plaintext $pt = \Delta m$

Sample $e \leftarrow \chi^n$  

Output ciphertext $ct \in \mathbb{Z}_q^{n+1}$ for plaintext $pt$ with secret key $sk$:
$$ct = (a_0, a_1, ..., a_{n-1}, b) | (a_0, a_1, ..., a_{n-1}) \leftarrow \mathbb{Z}_q, b = \sum_{i=0}^{n-1} a_is_i + e$$

## LWE to TLWE

Discretised TLWE is directly related to LWE when we identify elements in discretised $\mathbb{T}_q \subset \mathbb{T}$ (recall discretised torus as subset of real values in torus with precision $q$) with elements in $\mathbb{Z}_q$. 

Any element $a \in \mathbb{T}_q$ can be written as
$$a = q^{-1} a'$$ where $a' \in \mathbb{Z}_q$.


# Ring Polynomials

## Cyclotomic polynomials

$M^{th}$ cyclotomic polynomial is defined as:

$$
\Phi_M = \prod_{\zeta_i \in P(M)} X - \zeta_i
$$

where $P(M)$ is set of primitive $M^{th}$ roots of unity. 

The class of cyclotomic polynomials of interest to are where $M$ is some power of 2. 

Observe that due to the following theorem: 

For any positive integer $n >= 1$ and k s.t. $k | n$, 
$$
\Phi_{nk}(x) = \Phi_{n}(x^k) 
$$


We can re-write $M^{th}$ cyclotomic polynomial as: 
$$
\Phi_{M}(x) = \Phi_{M/2}(x^2) 
$$

Given M is some power of $2^m$, we can write:
$$
\Phi_{M}(x) = \Phi_{2}(x^{M-1})
$$
since, 
$$
\Phi_{2}(x) = X^2+1
$$
we can write M^th cyclotomic polynomial as
$$
\Phi_{M}(x) = \Phi_{2}(x^{M/2})
$$
Let $N = M/2$. Thus, 
$$
\Phi_{M}(x) = \Phi_{2}(x^{N}) = X^N+1
$$


## Why negacylic polynomials

The reason why we are interested in only cyclotomic polynomials that are some power of 2 is because they can be used to construct ring such as $Z_{N,q} = Z_q / X^N + 1$. Polynomials in such rings are known as negacylic polnyomials. 

> *Note*: 
> I don't know why do we use X^N+1 instead of X^N-1. But this can be because rings formed using [X^N+1 are considered](https://crypto.stackexchange.com/questions/100211/rlwe-explanation/100218#100218) [more secure than X^N-1](https://jeremykun.com/2022/12/09/negacyclic-polynomial-multiplication/).

One special thing about polynomials in ring $Z_{N,q}$ is that multiplications wraps around and when they do the coefficients are negated. 

For example, 
TODO

## LWE to RingLWE

Due to negacylic property of ring polynomial $Z_{N,q}$ polynomial multiplication between two polynomials $A, B \in Z_{N,q}$ and adding polynomial $E \in Z_{N,\chi}$ where $q$ is ciphertext modulus is equivalent to N LWE ciphertexts. 

TODO: show a diagram

Since one can calculate $A \cdot B$ in O(N log N) using NTT, encryption using RingLWE is more efficient than encrypting $N$ LWE ciphertexts. 

RingLWE packs plaintexts more tightly than LWE ciphertexts. For ex, to encrypt $N$ plaintexts RingLWE requires two polynomials (each with $N$ values $\in Z_q$), whereas LWE requires $N$ LWE ciphertexts.


# GLWE

Note: RLWE is equivalent to GLWE when GLWE dimension is set to 1

GLWE is parameterised $q,p,N,k, \sigma_g$. $Z_{N, q}$ is ciphertext polynomial ring and $Z_{N,p}$ is plaintext polynomial ring, and $k$ is GLWE dimension. 

## Encryption

To encrypt plaintext $pt \in Z_{N,p}$.

Sample secret key $sk = [s_0, ...,s_{k-1}]$ as list of $k$ polynomials $\in \mathbb{B}_{N, q}$. 

Sample $e$ as a polynomial $\in \mathbb{Z}_{N,q}$ with each coefficient sampled from distribution $\chi_{\sigma_g}$.

Output ciphertext $ct \in (\mathbb{Z}_{N,q})^{k+1}$ as
$$ct = (a_0, a_1, a_2,..., a_{k-1},b) | \space  (a_0, a_1, a_2,..., a_{k-1}) \leftarrow (\mathbb{Z}_q/(X^N + 1))^k, \space b = \sum_{i=0}^{k-1} a_is_i + e + pt$$

## Key Switching

Key Switching switches an LWE ciphertext encrypted under $s'$ to encryption under $s$.

Recall to decrypt an LWE ciphertext $ct = (a_0, ..., a_{n-1}, b)$ encrypted using s' one must perform:
$$\Delta(m) + e = b - \sum_{i=0}^{n-1}a_is'_i$$
To key switch $ct$ from an encryption under $s'$ to encryption under $s$, the idea is to calculate decryption procedure homomorphically. However doing so naively will require multiplying $a_i$ with LWE encryptions of $s'_i$ and, due to $a_i$ being in $Z_q$, will cause error in $LWE(s'_i)$ to blow up. Key switching procedure gets around this by decomposing $a_i$ using small base $\beta$ and encrypting corresponding $s_i$ scaled by recomposition factors in multiple LWE ciphertext.  

### Digit decomposition

Given a base $\beta$ and levels $l$ we construct the following gadget matrix: 
$$G = [\frac{q}{\beta},...,\frac{q}{\beta^l}]$$
Then we decompose $a \in Z_q$ as: 
$$G^{-1}(a) = [a_0,...,a_{l-1}]$$
where $a_0$ is most significant $log(\beta)$ bits of $a$, $a_1$ is next $\log{\beta}$, and so on and so forth.

Observer that we can reconstruct $a$ upto $l$ level accuracy as
$$a = G^{-1}(a)G^{T}$$
In practice we choose $\beta * l$ smaller than $q$ and safely ignore least significant bits of ciphertexts since they consist of noise. 

### Key Switching Key

Key switching requires a key switching key. Key switching key consist of LWE encryptions of bits in $s'$. 

Notice that $s'_i$ must be encrypted such that when its LWE ciphertexts are multiplied with vector $G^{-1}(a_i)$ of its corresponding $a_i$ it must result in LWE ciphertext encrypting $a_is_i'$. Thus we produce $l$ LWE ciphertexts of $s'_i$ as: 
$$[LWE(\frac{q}{\beta}s',s),...,LWE(\frac{q}{\beta^l}s', s)]$$

Notice that to encrypt we use a different LWE secret key $s$, which is the secret key we want to switch to.

Key switching key $ksk_{s' \rightarrow s}$ consist of $l \cdot n$ LWE ciphertexts. 

### Key Switching Procedure

With $ksk_{s' \rightarrow s}$ and LWE ciphertext $ct = (b, [a_0, ..., a_{n-1}])$, first decompose each $a_i$ using digit decomposition:
$$G^{-1}(a_i) = [a_{i,0},...,a_{i,l-1}]$$

Now calculate: 
$$LWE_s(m) = (0,...0, b') - \sum_{i=0}^{n-1} G^{-1}(a_i)\cdot ksk_{s' \rightarrow s}[i]^T$$

Where, 
$$
LWE_s(a_i, s_i) = G^{-1}(a_i)\cdot ksk_{s' \rightarrow s}[i]^T
$$
calculates inner product of $a_i$ and $s_i$ homomorphically.

Notice that since $a_{i,j}$ has norm of atmost $|\beta|$ it only scales error term in LWE ciphertext of secret bits by $|\beta|$ which is a lot smaller than norm of ciphertext modulus $q$.

> *Note*
> We use key switching in bootstrapping to switch LWE ciphertext extracted in sample extraction from encryption under $s' \in \mathbb{B}^{kN}$ (i.e. GLWE secret key interpreted as LWE secret key) to encryption under $s \in \mathbb{B}^n$. This implies $ksk_{s' \rightarrow s}$ can be viewed as 2D matrix $kNl$ rows with each row an LWE ciphertext. Thus, $ksk \in (\mathbb{Z}^{n+1}_q)^{(kN \cdot l)}$ 

# GGSW

One cannot multiply an GLWE ciphertext with another without blowing up the noise. 

Generally, in other schemes like BFV, BGV (where only RLWE is used) a bigger ciphertext modulus is used for noise accumulation which is further carefully kept under control. Instead TFHE uses flattening approach to multiply a masked message $m$ with a GLWE ciphertext.

The trick is to decompose GLWE ciphertext to reduce their norm and multiply it with a GGSW ciphertext (defined below) that recomposes polynomials in GLWE ciphertext, thus keeping noise under control.

## Gadget matrix

Let decomposition base be $\beta$ and let no. of decomposition level be $l$. We define a gadget matrix $G^T$ with dimension $(l(k+1)) \times (k+1)$ as:
$$G^T = \begin{pmatrix}
    \frac{q}{\beta} & \cdots & \cdots \\
    \vdots & \cdots & \cdots  \\
    \frac{q}{\beta^l} & \cdots & \cdots\\
    \vdots & \cdots & \cdots \\
    & \cdots  & \cdots \\
    & \ddots & \\
     \cdots & \cdots & \frac{q}{\beta} & \\
     \cdots & \cdots & \vdots   \\
    \cdots & \cdots& \frac{q}{\beta^l} & \\
\end{pmatrix}$$

Each element in $G^T$ is $\in Z_q$ and can be interpreted as a constant polynomial $\in Z_{N,q}$. Thus we can write $G^T \in \mathbb{Z}_{N,q}[X]^{l(k+1) \cdot (k+1)}$.

## GGSW Ciphertext

To encrypt message $m \in Z_{N,q}[X]$, we first calculate $mG^T$, then we add $mG^T$ to $\pi$. $\pi$ is a matrix of size $(l(k+1)) \times (k+1)$ with each row a zero GLWE encryption. 

$$\pi = \begin{pmatrix}
    GLWE_{sk}(0) \\
    GLWE_{sk}(0)\\
	\vdots \\
    GLWE_{sk}(0) \\
\end{pmatrix} = \begin{pmatrix}
    a_{0,0}, a_{0,1}, ..., a_{0,(k-1)}, b_0 \\
    a_{1,0}, a_{1,1}, ..., a_{1,(k-1)}, b_1 \\
	\vdots \\
    a_{(lk),0}, a_{(lk),1}, ..., a_{(lk),(k-1)}, b_{lk} \\
\end{pmatrix} \in \mathbb{Z}_{N,q}[X]^{(k+1)l \cdot (k+1)}$$

>Note
>$mG^T$ translates to multiplying each constant polynomial in gadget matrix $G^T$ with $m$

Since, 
$$mG^T = \begin{pmatrix}
    m\frac{q}{\beta} & \cdots & \cdots \\
    \vdots & \cdots & \cdots  \\
    m\frac{q}{\beta^l} & \cdots & \cdots\\
    \vdots & \cdots & \cdots \\
    & \cdots  & \cdots \\
    & \ddots & \\
     \cdots & \cdots & m\frac{q}{\beta} & \\
     \cdots & \cdots & \vdots   \\
    \cdots & \cdots& m\frac{q}{\beta^l} & \\
\end{pmatrix}$$

$$
\pi + mG^T = \begin{pmatrix}
    a_{0,0} + m\frac{q}{\beta}, a_{0,1}, ..., a_{0,(k-1)}, b_0 \\
    a_{1,0} + m\frac{q}{\beta^2}, a_{1,1}, ..., a_{1,(k-1)}, b_1 \\
	\vdots \\
    a_{(lk),0}, a_{(lk),1}, ..., a_{(lk),(k-1)}, b_{lk} + m\frac{q}{\beta^l} \\
\end{pmatrix}
$$


## External product

Given $GLWE_{sk}(m_1)$ and $GGSW_{sk}(m_2)$ one can calculate their external product to produce $GLWE_{sk}(m_1m_2)$.

First, decompose $GLWE_{sk}(m_1) = (a_0, a_1, ..., b)$ to produce $l \cdot (k+1)$ polynomials as $$G^{-1}(GLWE_sk(m_1))=[a_{0,0}, a_{0,1}, ..., a_{0,l-1}, ..., b_{0}, ..., b_{l-1}]$$
> *Note*
> chunk $[a_{i,0}, a_{i,1}, ..., a_{i,l-1}]$ is decomposition of polynomial $a_i$

Then calculate the external product as: 
$$
\\ G^{-1}(GLWE_{sk}(m_1)) \cdot GGSW_{sk}(m_2)
\\ = G^{-1}(GLWE_{sk}(m_1)) \cdot \pi + G^{-1}(GLWE_{sk}(m_1)) \cdot m_2G^T \\
\\ = 0 + GLWE_{sk}(m_2m_1) = GLWE_{sk}(m_2m_1)
$$


### Why does it work?

Think of external product as:
$$
[a_{0,0}, a_{0,1}, ..., a_{0,l-1}, ..., b_{0}, ..., b_{l-1}] \times \begin{pmatrix}
    a'_{0,0} + m_2\frac{q}{\beta}, a'_{0,1}, ..., a'_{0,(k-1)}, b_0 \\
    a'_{1,0} + m_2\frac{q}{\beta^2}, a'_{1,1}, ..., a'_{1,(k-1)}, b_1 \\
	\vdots \\
    a'_{(lk),0}, a'_{(lk),1}, ..., a'_{(lk),(k-1)}, b'_{lk} + m_2\frac{q}{\beta^l} \\
\end{pmatrix}
$$

The $i^{th}$ column, where $i \lt k$, calculates the inner product: 
$$
\sum_{j=il}^{i(l+1)-1}(a_{i,j}a'_{j,i} + a_{i,j}m_2\frac{q}{\beta^{j+1}}) \\
+ \sum_{j=0, j\neq[il, i(l+1)-1]}^{l\cdot(k-1)-1} a_{j/l,j \mod l}a'_{j,i} + a_{0,j}m\frac{q}{\beta^{j+1}}) \\ 
+ \sum_{j=l\cdot(k-1)-1}^{l\cdot(k)-1} b_{j \mod l}a'_{j,i}
$$

The latter two parts of the summation sum to zero encryption. The first part consist of zero encryption (i.e. $\sum a_{i,j}a'_{j,i}$) and recomposes $a_{j}$ while multiplying it with $m_2$ (i.e $\sum m_2\cdot a_{i,j}\frac{q}{\beta^{j+1}})$). This results in polynomial $i^{th}$ mask component of resulting GLWE encryption of $m_1m_2$.

Same applies for the last column which recomposes $b$ component of $GLWE(m_1)$ while multiplying it with $m_2$, thus resulting in $b$ component of new GLWE ciphertext encrypt $m_1m_2$.


## GLEV and GGSW

Given message $m_1 \in Z_{N, q}$, and decomposition parameters base \beta, and level l, $GLEV(m_1)$ is: 
$$GLEV(m_1) = [GLWE(m_1\frac{q}{\beta^{1}}),..., GLWE(m_1\frac{q}{\beta^{l}})] \in R_q^{l \times (k+1)}$$

Notice that GLEV encryption encrypts $m_1$ with different recomposition factors, starting with recomposition factor for most significant digits. 

**GGSW encryption** of $m \in Z_{N,q}$ now becomes a collection of GLEV encryptions as: 
$$GGSW(m) = [GLEV(-S_i\cdot m), ..., GLEV(-S_{k-1} \cdot m), GLEV(m)] \in R_q^{(k+1) \times (l(k+1))}$$
where $S_i$ is $i^{th}$ secret polynomial in GLWE secret key $sk$.

**External product**

Given $GLWE(m_1)$ and $GGSW(m_2)$, we first calculate $G^{-1} (GLWE(m_1))$ as:
$$G^{-1} (GLWE(m_1)) = G^{-1} [a_0, ..., a_{k-1}, b] = [a_{0,1}, ..., a_{0,l}, a_{1,1}, ..., a_{k-1,l}, b_{1},..., b_{l}]$$
 Then we calculate inner product between $G^{-1} (GLWE(m_1))$ and $GGSW(m)$ as:
 $$\sum_{i=0}^{k-1}<[a_{i,1}, ... a_{i,l}] GLEV(-S_im_2) > + <[b_{1}, ... b_{l}] GLEV(m_2) >$$
Since $[a_{i,1}, ... a_{i,l}]$ are decomposed values of $a_i$ (starting with most significant digit), 
$$<[a_{i,1}, ... a_{i,l}] GLEV(-S_im_2) > \space = \sum_{j=1}^{l} a_{i,j} GLWE(-Sm_2\frac{q}{\beta^{j+1}})$$
$$= GLWE(-S_ia_im_2)$$
Thus since $b - \sum S_ia_i = \Delta m_1$, 
$$=\sum GLWE(-S_ia_im_2) + GLWE(bm_2) = GLWE(\Delta m_1m_2 )$$

## GLEV approach vs Normal Approach

The only good reason that I am able to come up with for using GLEV approach over normal approach is that GLEV requires $l(k+1)$ multiplication in $Z_{N,q}$ whereas normal approach requires $l(k+1) \times (k+1)$ multiplications (normal approach multiplies row vector with a matrix).




# CMUX

Once we have GGSW ciphertext, CMUX operation becomes really easy to define. The core idea of cmux is to calculate 2 input multiplexer homomorphically as: 
$$c \cdot (m_1-m_2) + m_2$$
where $c$ is indicator bit and $m_1, m_2$ are messages. 

Define $c \in [0,1]$ and encrypt it as a GGSW ciphertext: $GGSW_{sk}(c)$. Given two GLWE ciphertexts $GLWE_{sk}(m_1)$ and $GLWE_{sk}(m_2)$ calculate: 
$$m = GLWE_{sk}(m_1) - GLWE_{sk}(m_2)$$
$$out = \text{ExternalProduct}(GGSW_{sk}(c),m) + GLWE_{sk}(m_2)$$
Since $c$ can either be 0 or 1, $out$ will be GLWE encryption of $m_1$ if $c=1$, otherwise $m_2$.



# Bootstrapping

Bootstrapping takes in a noisy LWE ciphertext and homomorphically runs the decryption procedure to product a fresh LWE ciphertext with fixed (& reduced) noise. 



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

## Intuition

Recall that decryption of LWE ciphertext is defined as:
$$\mu = \Delta{m} + e = b - \sum{a_is_i} \mod{q}$$

Now the question is, given that LWE, GLWE, and GGSW exist can we somehow calculate \mu homomorphically? And yes we can. 

Recall that a plaintext in GLWE is $\in Z_{N,q}$ and we can multiple a GLWE ciphertext with a GLWE plaintext. The trick is to calculate \mu in the exponent of a GLWE plaintext. 

First, let's look at how it is done using a plaintext.

Given a LWE ciphertext $ct = (a_0, a_1, ..., a_n, b)$. Define $X \in Z_{N,q}$. Then raise $X$ to $-b$. Thus producing a $X^{-b}$. Then given LWE secret vector (s_0, s_1, ..., s_n) we calculate the inner product $\sum a_i\cdot s_i$ in the exponent and multiply the resulting polynomial with $X^{-b}$ thus producing a polynomial as $X^{-b + \sum a_is_i}$: 
$$
X^{-\mu} = X^{-b + as} = X^{-b}\prod_{i=0}^{n-1} (X^{a_is_i})
$$

Now if we construct another polynomial $v(x)$ such that it encodes the desirable corresponding plaintext in coefficient $\mu$, then $v(x) X^{-\mu}$ will output a polynomial with the desired value in constant term (i.e. $X^{-\mu}$ rotates $v(x)$ left by $\mu$). 

## Switching modulus from q to 2N

One thing we ignored above is that $-\mu$ is calculated $\mod 2N$ instead of $\mod q$. The reason for this is that polynomial X $\in Z_{N,q}$ is negacylic thus has an order of $2N$ ($X^{2N} = 1$). 

To switch the modulus from q to 2N we calculate:

$$b' = round(\frac{2N}{q}b) \mod 2N$$
$$a_i' = round(\frac{2N}{q}a_i) \mod 2N$$
>Note:
>Switching modulus introduces additional error to bootstrapping procedure. This error is known as *drift*. One can reduce its impact over bootstrapping procedure by setting correct parameters. 


## Constructing $v(x)$ - test polynomial

To construct v(x) correctly, let's look at possible values of $\mu$. 

$\mu = \Delta m + e$

Since for any plaintext m \mu can be several values due to $e$, we cannot encode m in just one of the coefficients of v(x). Instead we will have encode it in a continuous block of coefficients. Considering this and the fact the $\mu$ is periodic (i.e. block for plaintext 0, comes before block of plaintext 1 and so on and so forth), we can deduce the following: 
1. Let's divide coefficient vector of v(x) into sets of p blocks, each with $v(x) / p$ coefficient slots. 
2. The encode plaintext value of 0 in first block, plaintext value of 1 in second block, and so on and so forth. 

For ex, if plaintext modulus $p = 4$ and $v(x)$ is of degree 15 then coefficient vector has 4 blocks each with 4 slots and can be constructed as:

$$[0,0,0,0,\Delta1, \Delta1, \Delta1, \Delta1, \Delta2,\Delta2,\Delta2,\Delta2,\Delta3,\Delta3,\Delta3,\Delta3]$$

However, notice that $e$ can be negative, so starting the first block at the first coefficient does not matches the periodicity of plaintext over torus. Instead it is exactly rotated to right by half of the size of block. To match the periodicity of torus we rotate the vector left by half of the size of block. 

$$[0,0,\Delta1, \Delta1, \Delta1, \Delta1, \Delta2,\Delta2,\Delta2,\Delta2,\Delta3,\Delta3,\Delta3,\Delta3, 0,0]$$

But there's another issue. Notice what happens when error is -ve and plaintext is 0. The multiplication wraps arounds and brings some part of half window corresponding to 0 from top coefficients to lowest coefficients, thus negating the terms. To avoid resulting value to be -ve of what it must be, for the half window stored in top coefficient we should encode their negative representations.

$$[0,0,\Delta1, \Delta1, \Delta1, \Delta1, \Delta2,\Delta2,\Delta2,\Delta2,\Delta3,\Delta3,\Delta3,\Delta3, -0,-0]$$

Notice that uptill now we encoded the identity function. That is, for any given plaintext we encoded the plaintext itself, $f(x) = x$. Moreover, we can change the function to any arbitrary function $f(x)$ that maps input plaintext to output plaintext. This is known as programmable bootstrapping (PBS). 

>Note
> $f(x)$ maps any input plaintext to any output plaintext as long the parameters are suitable with output plaintext. This implies, with bootstrapping one can change the plaintext space as well. 


However, we still aren't done. Notice that $\mu$ can be in range $0 < \mu < N$ or in range $N <= \mu < 2N$. If it is former, then our multiplication by $v(x)$ works as expected and we obtain correct encoded plaintext value as constant term. However if it is in latter range then we obtain -ve of $k^{th}$ coefficient where $k = \mu - N$. This is because for $\mu > N$, $v(x)$ wraps around thus negating the terms. One way to avoid this is to force $\mu$ to be in range $0 <= \mu < N$ and doing requires reserving a bit of plaintext as a padding bit. Thus, after padding bit, the plaintext is restricted to half of what is was before. 

> Note
> With padding bit $\mu$ is not always restricted in range $0 <= \mu < N$. It may be -ve, but only when plaintext corresponds to 0 and we accommodate for that by encoding half window corresponding to plaintext 0 in top slots. 


> Note
> Multiplication $X^{-\mu}v(x)$ can be generalised as
> $X^{-\mu} \cdot v(x) = (-1^{\text{floor}(\mu/N)}) \cdot X^{-(\mu \mod N)}$

## Negacylic functions

These are class of functions such that $f(x+\frac{p}{2}) = -f(x)$. 

For ex, let $p = 2$ . Notice that function $f(x) = x$ is negacylic. 

The benefit of negacylic functions is that, unlike general functions, one does not need to reserve padding bit. The reason for this is when $\mu$ is in range $N <= \mu < 2N$ the resulting constant term coefficient is -ve of $k^{th}$ coefficient where $k + N = \mu$. But if -ve of k^th coefficient is the desired value, then $X^{-b}v(x)$ result into polynomial with expected plaintext value in constant term. This will be the case when any plaintext maps to a value that is negative of what a plaintext halfway across it maps to (Notice that $N$ equates to $p/2$ in plaintext space).

>TODO
>Explanation above becomes clearer with torus. Add one here.

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

For ciphertext modulus $q$, base $\beta$ and level $l$, we represent gadget vector as:
$$g = [\frac{q}{\beta}, \frac{q}{\beta^2} ..., \frac{q}{\beta^{l}}]$$
> Note 
> When $\beta * l = q$ we can write $g$ as: $g = [\beta^{l-1},..., 1]$

Given an unsigned integer $a \in Z_q$ we can write its decomposition vector of size $l$ as: 
$$g^{-1}(a) = [a_0, a_1, ..., a_{l-1}] \in Z_{\beta}^l$$
where we start decomposition with MSBs. This implies $a_0$ is $log(\beta)$ MSBs of $a$.

To recompose $a$, we may calculate: 
$$g^{-1}(a)g \approx \sum_{i=0}^{l-1} a_i(\frac{q}{\beta^i})$$

> *Note* 
> Recomposition is exact when $\beta * l = q$

## Why Decomposition

Decomposition is used to decrease noise growth when multiplying a ciphertext with another value in $Z_q$.

For example, consider the structure of $b^{th}$ component of LWE ciphertext: 
$$
b = a\cdot s + e +  m_1
$$
if we were to multiply $m_2 \in Z_q$ with $b$ then we will scale the error $e$ by norm of $q$, thus causing error to blow up and making the ciphertext useless. 

On the other hand, one can reduce the norm of $m_2$ by decomposing it into smaller values. So if one can somehow re-compose $m_2$ while multiplying it with ciphertext(s) encrypting $m_1$, once can produce ciphertext encrypting $m_1m_2$ without causing noise to blow up. 

Turns out, this isn't doable with single ciphertext of $m_1$. Instead mutliple redundant ciphertext of $m_1$ are required. 

The idea is to first paramterize decomposition base \beta and level l. Then produce $l$ redundant LWE ciphertexts of $m_1$ as:
$$LWE_s(g(m_1)) = [LWE_s(m_1\frac{q}{\beta}), LWE_s(m_1\frac{q}{\beta^2}) ..., LWE_s(m_1\frac{q}{\beta^{l}})]$$
Then decompose $m_2$ and produce gadget vector $g(m_2)$:
$$g^{-1}(m_2) = [m_{2,0}, m_{2,0}, ..., m_{2,l-1}] \in Z_{\beta}^l$$

Then calculate:
$$LWE(g(m_1))^T \cdot g^{-1}(m) = \sum_{i=0}^{l-1} m_{2,i}LWE_s(m_1\frac{q}{\beta^{i+1}})$$
Since,
$$
m_{2,i}LWE_s(m_1\frac{q}{\beta^{i+1}}) = LWE_s(m_{2,i}m_1\frac{q}{\beta^{i+1}})
$$

$m_{2,i}$ is scaled by it's corresponding recomposition factor and summation results in:
$$LWE_s(m_1m_2)$$

>Note
>Since norm of $m_{2,i}$ is in worst case equal to norm of $\beta$ the noise grows only by small amount.


### Decomposition for GLWE

Same trick to reduce noise growth in LWE setting can be applied to GLWE setting. The only difference is in GLWE setting decomosition of a ring polnomial in $Z_{N,q}$ translates to decompsing its coefficients in $Z_q$.

For ex, decomposition of polynomial $a \in Z_{N,q}$ equals $l$ polynomials in $Z_{N,q}$

## Types of decomposition
### Unsigned decomposition

Unsigned decomposition is what you will normally think when decomposing an unsigned integer. For ex, decomposition of $a \in Z_q$ is as:
$$g^{-1}(a) = [a_0, a_1, ..., a_{l-1}] \in Z_{\beta}^l$$
starting with MSBs of $a$.
### Signed Decomposition

> Note
> Signed decomposition is preferred method in practice

Signed decomposition further improves upon unsigned decomposition in terms of noise growth. Instead of each element of decomposed vector being $\in [0, \beta)$ as in unsigned decomposition, each element is $\in [-\beta/2, \beta/2)$ (i.e. absolute value is 1/2 of before). 

To construct signed decomposition of $a \in Z_q$ the idea is to convert decomposed values to their signed representation, but we need to take care of carry overs. For example, let decomposition of $a$, starting from LSB, be $[a_0, a_1, ..., a_{l-1}]$. To obtain signed decomposition such that each value $a_i$ is converted to its signed representation $\in [-\beta/2, \beta/2)$ we subtract $\beta$ from $a_i$ if $a_i \geq \beta/2$ . If $a_{i}$ is subtracted then we must add (ie carry over) 1 to $a_{i+1}$ and then convert $a_{i+1}$ to its signed representation. This works because: 
$$a_{i-1} \beta^{i-1} + a_{i} \beta^{i} = a_{i-1} \beta^{i-1} - \beta^i + a_{i} \beta^{i} + \beta^i = \beta^{i-1}(a_{i-1} - \beta) + \beta^{i}(a_i + 1)$$

The benefits of using signed decomposition are: 
1. each value $a_i$ in its signed representation is in $\in [-\beta/2, \beta/2)$ and has maximum possible magnitude $\beta/2 - 1$. This means noise growth is half of what it will be in unsigned decomposition. 
2. The maximum representable value, i.e. $a$, is reduced to $\sum (\beta/2 - 1)\beta^i$. This means we need to limit the value of $a$. TODO: I am unsure how is this handle in practice? Setting padding bit to 1 may handle, but what about the cases where don't want reserver a bit of padding?

For more information on signed decomposition please refer to this [blogpost](https://jeremykun.com/2021/12/11/the-gadget-decomposition-in-fhe/).


# Parameters and noise analysis

## Swapping order of bootstrapping in and keyswitching

As highlighted in the [guide](https://assets.researchsquare.com/files/rs-2841900/v1_covered.pdf) and first observed in [paper](https://eprint.iacr.org/2017/1114.pdf) and further analysed in [paper](https://eprint.iacr.org/2022/704), swapping the order of key switching with bootstrapping results lesser noise growth during evaluation of TFHE operations, thus more efficient parameters. 

In the originally proposed version of TFHE, a given LWE ciphertext $\in Z_q^{n+1}$ is first bootstrapped and then key switched from LWE secret key $s' \in \mathbb{B}^{k \cdot N}$ (ie GLWE key viewed as LEW key) to $s  \in \mathbb{B}^{n}$ (ie LWE secret key using which the input ciphertext is encrypted). It outputs LWE ciphertext $\in Z_q^{n+1}$. 

To swap the order, the input LWE ciphertext $\in Z_q^{k \cdot N + 1}$ is first key switched from $s'$ to $s$. Then bootstrapped to output a ciphertext $\in Z_q^{k \cdot N + 1}$. To gain an intuitive understanding of why switching the order reduces noise during TFHE gate evaluation, consider the following gate: 

Let's define a gate $G$ that given LWE ciphertext $ct_0, ct_1, ..., ct_n$ outputs a ciphertext $ct_s$  such that
$$m_s = \sum w_i m_i$$
The gate calculates: 
$$ct_r = \sum w_i \cdot ct_i$$
For some weights $w_0, w_1,...,w_n \in Z_p^{n}$ 

Assuming the noise variance of $ct_i$ as $v_i$, the noise of output ciphertext will be weighted sum squares of $||w_i||$: 
$$v_s = \sum ||w_i||^2 \cdot v_i$$
Assuming variance of all input ciphertexts $ct_i$ equals $v_t$ we can re-write $v_s$ as:
$$v_s = v_t \sum ||w_i||^2 $$

In originally proposed version of TFHE, $v_t$ will equal: 
$$v_t = v_{bs} + v_{ks}$$
where $v_{bs}$ is noise in refreshed ciphertext after bootstrapping and $v_{ks}$ is noise incurred in key switching. 

However, in swapped version of TFHE $v_t$ will equal $v_{bs}$. Thus during gate evaluation noise growth will be $v_{bs} \cdot \sum ||w_i||^2$ as opposed to $(v_{bs} + v_{ks})\sum ||w_i||^2$ in the original version. 

If one were to evaluate the gate $G$ and bootstrap repeatedly in a cycle, then noise growth in one cycle, before bootstrapping again, will be: 

Orignal version: 
$(v_{bs} + v_{ks}) \cdot ||w_i||^2 + v_{drift}$

New version:
$(v_{bs}) \cdot ||w_i||^2 + v_{ks} + v_{drift}$

Where $v_{drift}$ is additional noise due to modulus switching $q \rightarrow 2N$

# Rough points: 

1. To compare PBS accumulation step performance of different parameter sets you can:
	1. Estimate size of GGSW:
	   as $(k + 1) \cdot (l \cdot (k+1))$ elements $\in R_q$
	    Thus, $(k + 1) \cdot (l \cdot (k+1)) \cdot 2^N$ elements $\in Z_q$
	2. Estimate cost of converting decomposed GLWE ciphertext to fourier domain as: 
	   $l \cdot (k + 1) \cdot(N \log{N})$
	3. Since a single $GGSW \times GLWE$ requires as many as there are $Z_q$ elements in GGSW - $(k + 1) \cdot (l \cdot (k+1)) \cdot 2^N$ multiplications and a single PBS accumulation performs $n$ $GGSW \times GLWE$ operations, total cost of PBS roughly equals: 
	   $(k + 1) \cdot (l \cdot (k+1)) \cdot 2^N$ + $l \cdot (k + 1) \cdot(N \log{N})$ 
	   
Note that $l$ is levels in PBS and $n$ is lwe dimension. 

Thanks @icetdrinker for pointing out the formulas for estimating costs. 

2. 





TFHE-rs TODO
1. Under polynomial karatsuba multiplication [here](https://github.com/zama-ai/tfhe-rs/blob/80b5ce7f63a985be2a7d3bc636729e1c8e24a334/tfhe/src/core_crypto/algorithms/polynomial_algorithms.rs#L549).
2. Replace definitions of base decompositions such that we don't have to assume $\beta l = q$.
3. 



---
Key switching procedure

%% LWE encryption of each bit of $s'$. To use digit decomposition and recomposition during key switching we decompose each bit in $s'$ and encrypt them. That is, 
1. Decompose $s'_i$ using digit decomposition. Since $s_i$ is a bit, in practice this is omitted. 
2. Given gadget row vector $G$, calculate $s'_iG = [s'_i\beta^{l-1}, ..., s'_i]$.
3. Encrypt each element of $s'_iG$ using $s$. This produces $l$ LWE ciphertexts. 
After encrypt all bits in $s'$ we obtain an 2D array of LWE ciphertext where $j^{th}$ column of $i^{th}$ row is an LWE encryption as $LWE_s(s'_i\beta^{l-{j+1}})$. Let's call this 2D array key switching key $ksk$.

We then perform key switching for LWE encryption $LWE_{s'}(m) = (a'_0,...,b')$ as
$$LWE_s(m) = (0,...0, b') - \sum_{i=0}^{n-1} G^{-1}(a_i')ksk[i]^T$$
Notice that $G^{-1}(a_i')ksk[i]^T$ expands to 
$$[a'_{i0},...,a_{i(n-1)}'][LWE_{s}(s'_i\beta^{l-1}),...,LWE_{s}(s'_i)]^T = LWE(a'_is_i)$$
but without accumulating large noise since magnitude of each scalar value $a'_{il}$ is limited to base $\beta$. %%

**Decompsition**


For example, let $ct = LWE_s(m_1) = (a_0,...,b)$ and we would want to multiply $m_2 \in Z_q$


We can multiply $c$ by $ct$ to produce $LWE_s(cm)$, that is a ciphertext encrypting product $cm$. However notice that after multiplication noise in the ciphertext grows since: 
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

**External product**
Then 

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


## Ciphertext Multiplication

You can easily multiply a GLWE ciphertext with a scalar by multiplying polynomial with the scalar. However, multiplications between two ciphertexts is not trivial. This is because internal product over Torus is not defined, you need to use external product. 

Let $x \in \mathbb{Z}$ and $a \in \mathbb{T}$. External product between $x$ and $a$ is defined as
$$x \cdot a = a + a + ... + a \space (x \ times)$$

We cannot port this definition directly to encrypted context. In other words, we can cannot add a given GLWE ciphertext by itself $x$ no. of times, where $x$ itself is unknown. To perform ciphertext multiplication between $x \in \mathbb{Z}_{N}[X]$ and $a \in \mathbb{Z}_{N,q}[X]$ we must first encrypt $a$ under GLWE and $x$ under GGSW. Result of their external product equals GLWE encryption of $x \cdot a$, that is 
$$GLWE(x \cdot a) = GGSW(x) \cdot GLWE(a)$$

%%To clear any confusion recall that we view an element defined over discretized torus through its representation in $\mathbb{Z}_q$. That is $a \in \mathbb{Z}_{N,q}[X]$ is defined over $\mathbb{T}_{N,q}[X]$ as 
$$a' = q^{-1}\mathbb{Z}_{N,q}[X]$$%%


**Extra boostrapping** 
During bootstrapping we construct a polynomial as: 
$$X^{-b + \sum{a_is_i}}$$
and then multiply it with test polynomial $v(x) \in R_q$, that is
$$X^{-\mu} \cdot v(x)$$
The output polynomial rotates $v(x)$ by $-\mu$. Test vector polynomial $v(x)$ is constructed such that desired plaintext value is obtained in constant term of output polynomial after rotation. 

> **Note**
> Since $R_q = Z_q/X^N+1$, polynomial has order $2N$ and $X^{-\mu} = X^{2N - \mu}$

Notice that if $0 \leq \mu \leq N$, then $X^{-\mu} \cdot v(x)$ outputs $\mu^{th}$ coefficient of $v(x)$. 

However when $N \leq \mu \leq 2N$, $X^{-\mu} \cdot v(x)$  outputs $-(\mu - N)^{th}$ coefficient of $v(x)$. This is because $\mu = N + k$ and due to negacyclic property of $R_q$ multiplication by $X^N$ wraps around and negates the coefficients. 

To generalise: 
$$X^{-\mu} \cdot v(x) = (-1^{\text{floor}(\mu/N)}) \cdot X^{-(\mu \mod N)}$$

Moreover, observe that in $v(x)$ we can only encode $N$ elements and the rest are obtained as negative of the encoded elements. This implies bootstrapping natively suits only a certain class functions: $f(x + p/2) = -f(x)$ (where p is input space; plaintext space). Such functions are called negacyclic functions. (Note:  $p$ and $N$ are interchangeable in context of the function $f(x)$ because LWE ciphertext is modulus switched from $q \rightarrow 2N$ as $\text{round}(2N \cdot ct) \mod 2N$ to maintain periodicity).

To encode general functions in test polynomial (& identity function itself) it must be assured that $\mu \leq N$. In practice this is achieved by setting the first bit of plaintext space to 0. This has a consequence that plaintext space is now halved and there are solutions proposed, such as WoPBS, to get around this problem. 

Constructing test vector
In practice test polynomial is constructed as (we view polynomial with coefficient vector): 
1. Construct vector $m$ consisting of all possible values in message space $[0, 2^M)$, as:
   $$[0,..,2^M-1]$$
2. Divide N into $b$ blocks where $b = N/(2^M)$ and fill each block with repeated values of values in $m$ (this corresponds to mapping same noiseless values over the error range). 
   $$[0,0,0,0,1,1,1,1,...,2^M-1]$$ where $b = 4$.
3. Slice half of block corresponding to 0, negate its values and append it at the end. 
4. Encode the resulting coefficient as a plaintext, since we view them as a valid GLWE ciphertext. This means multiply each value by $\Delta$.

Note that replacing vector $m$ with an output of a function, turns bootstrapping into programmable bootstrapping (PBS). That is for function $f(x) | x \in [0, 2^M)$, construct $m$ as
$$m = [f(0), f(1), ..., f(2^M-1)]$$

It is worth noting that bootstrapping is PBS with identity function. 
