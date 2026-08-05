# DiTango

`chitu_diffusion.ditango` is the single canonical DiTango implementation.
It uses CuBic-style anchor-only curvature planning over CP groups.

## Solution: Attention State Reuse

To leverage this system-pattern alignment while addressing the memory challenge, we introduce \emph{attention state}, commonly used in partitioned parallel attention computation \cite{bpt,ringattention,flashattn2,flashinfer}, as our optimization medium. For sequence partition $i$, an attention state $\text{AS}_t(i)$ at timestep $t$ comprises output $\mathbf{OUT}_t(i)$ and log-sum-exp $\mathbf{LSE}_t(i)$:

$$
\text{AS}_t(i) = \begin{bmatrix}
\mathbf{OUT}_t(i) \\
\mathbf{LSE}_t(i)
\end{bmatrix}, \quad
\begin{aligned}
\mathbf{LSE}_t(i) &= \log \sum_{j \in \mathcal{I}_i} \exp(\mathbf{q}_t \cdot \mathbf{k}_j) \\
\mathbf{OUT}_t(i) &= \sum_{j \in \mathcal{I}_i} \frac{\exp(\mathbf{q}_t \cdot \mathbf{k}_j)}{\exp(\mathbf{LSE}_t(i))} \mathbf{v}_j
\end{aligned}
$$

where $\mathbf{q}$, $\mathbf{k}$, $\mathbf{v}$ are query, key, and value vectors.

A key property of attention states is their \textbf{composability}—they can be composed associatively and commutatively. For partitions $i$ and $j$:

$$
\text{AS}_t(i) \oplus \text{AS}_t(j) = \begin{bmatrix}
\frac{e^{\mathbf{LSE}_t(i)} \mathbf{OUT}_t(i) + e^{\mathbf{LSE}_t(j)} \mathbf{OUT}_t(j)}{e^{\mathbf{LSE}_t(i)} + e^{\mathbf{LSE}_t(j)}} \\
\log(e^{\mathbf{LSE}_t(i)} + e^{\mathbf{LSE}_t(j)})
\end{bmatrix}
$$

This composability enables flexible partition-wise computation and reuse strategies.