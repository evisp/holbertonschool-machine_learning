To prove that the sum of the squares of the first \( m \) positive integers is given by the formula:

\[
\sum_{k=1}^m k^2 = \frac{m(m+1)(2m+1)}{6},
\]

we provide a proof sketch using mathematical induction.

---

### **Base Case**
For \( m = 1 \):
\[
\sum_{k=1}^1 k^2 = 1^2 = 1,
\]
and the formula gives:
\[
\frac{1(1+1)(2\cdot1+1)}{6} = \frac{1 \cdot 2 \cdot 3}{6} = 1.
\]
Thus, the base case holds.

---

### **Inductive Step**
Assume the formula holds for \( m = n \), i.e., assume:
\[
\sum_{k=1}^n k^2 = \frac{n(n+1)(2n+1)}{6}.
\]

We need to prove it holds for \( m = n+1 \), i.e., that:
\[
\sum_{k=1}^{n+1} k^2 = \frac{(n+1)(n+2)(2(n+1)+1)}{6}.
\]

Start with the left-hand side:
\[
\sum_{k=1}^{n+1} k^2 = \sum_{k=1}^n k^2 + (n+1)^2.
\]

Using the inductive hypothesis:
\[
\sum_{k=1}^n k^2 = \frac{n(n+1)(2n+1)}{6},
\]
so:
\[
\sum_{k=1}^{n+1} k^2 = \frac{n(n+1)(2n+1)}{6} + (n+1)^2.
\]

Factor out \( n+1 \) from the two terms:
\[
\sum_{k=1}^{n+1} k^2 = \frac{(n+1)\left[n(2n+1) + 6(n+1)\right]}{6}.
\]

Simplify the expression inside the brackets:
\[
n(2n+1) + 6(n+1) = 2n^2 + n + 6n + 6 = 2n^2 + 7n + 6.
\]

Factor \( 2n^2 + 7n + 6 \):
\[
2n^2 + 7n + 6 = (n+2)(2n+3).
\]

Thus:
\[
\sum_{k=1}^{n+1} k^2 = \frac{(n+1)(n+2)(2n+3)}{6}.
\]

This matches the formula for \( m = n+1 \):
$$
\frac{(n+1)(n+2)(2(n+1)+1)}{6}.
$$

---

### **Conclusion**
By induction, the formula:
\[
\sum_{k=1}^m k^2 = \frac{m(m+1)(2m+1)}{6}
\]
is true for all \( m \geq 1 \).