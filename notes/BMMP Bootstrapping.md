
BMMP bootstrapping procedure unrolls the loop of external product such that number of external products are halved at the cost of increasing the bootstrapping key size. 

Recall that in bootstrapping blind rotate calculates $X^{-b + \sum a_is_i}v(x)$ as 
```
let acc = X^{-b}v(x)
for i in 1..n 
	// cmux gate
	acc = external_product(bk_{si}, acc(X^{a_i}) - acc) + acc
```
Notice that the above procedure require $n$ external products (external products are expensive operations)

BMMP reduces external products by half by noticing: 

$X^{a + a'} = ss'(X^{a+a'} - 1) + s(1 - s')(X^{a} - 1) + (1-s)s'(X^{a'}- 1) + 1$

For any $a, a' \in [1,n]$

You can plug in all possible inputs of $s$ and $s'$ (that is 0,0 OR 0,1 OR 1,0 OR 0,0) and check that equation holds. 

Using the decomposition above we can group calculating external product by each consecutive 2 terms, $X^{a}$ and $X^{a+1}$, into a single term $X^{a+a'}$, thus reducing number of external products by half. However, calculating the term $X^{a+a'}$ requires 3 GGSW scaling by plaintext polynomial, 2 GGSW additions and 1 GLWE addition (we assume that we take out 1 and instead add result with Acc). 

Bootstrapping key size is 1.5 times larger than bootstrapping key size required in normal bootstrapping. Bootstrapping keys $bk$ for BMMP is generated in sets of 3 as 
$$bk_{3i} = s_{2i}s_{2i+1}, bk_{3i+1} = s_{2i}(1-s_{2i+1}), bk_{3i+2} = s_{2i+1}(1-s_{2i})$$
for $i \in [1..(n/2)]$.


Referece: 
1. https://eprint.iacr.org/2017/1114.pdf
2. Jaxite implements BMMP bootstrapping https://github.com/google/jaxite
3. 