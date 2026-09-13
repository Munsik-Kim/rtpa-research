# Direction, correlation, and recurrent storage error

These are standard finite-dimensional identities and deliberately small CPU
counterexamples. They motivate measurements; they do not explain a measured
language-model KL ratio, prove DIAG optimality on the model, or establish novelty.
The [source record and synthetic preregistration](../data/benchmarks/grid_feedback/theory_sources.json)
distinguish primary text actually read from bibliography-only verification.

## A write affects later readouts

For fixed operands, let the stored-state difference satisfy

`delta_t = A_t delta_(t-1) + epsilon_t`, with `delta_(-1) = 0`.

The implementation updates the state, reads it, then quantizes storage. Thus the
readout difference at token `t` is `r_t = q_t^T A_t delta_(t-1)`, not
`q_t^T delta_t`. For one value column, stacking all writes gives

`r = L e`, with `L[t,j] = q_t^T A_t ... A_(j+1)` for `j < t`, and zero otherwise.

For several value columns, use the corresponding vectorization or sum column
energies. Token/window weights can be absorbed as square-root weights in L.
The diagonal write/readout block is zero. The final write is unread if no later
readout is scored. This timing convention is tested independently by direct
recurrence and a dense L, including nonsymmetric, noncommuting transitions.

Writing `r = sum_j r_j` gives

`J = ||r||² = sum_j ||r_j||² + 2 sum_(i<j) <r_i,r_j>`.

The cross-write terms can have either sign. Counting positive terms alone does
not give an effect-size threshold, and zero mean does not force half the terms
to be positive. A stored-grid repeat experiment, a frozen-operand recurrence,
and an actual model trajectory are different interventions.

For a fixed L and a finite-second-moment random error vector with mean `mu` and
covariance `Sigma`, expansion of the quadratic form gives

`E[J] = ||L mu||² + tr(L Sigma L^T)`.

No independence assumption is needed for this identity. It is invalid to replace
Sigma by its time-block diagonal unless the omitted covariance contribution is
zero or an explicitly justified approximation. If L is random and depends on
the same errors, the unconditional formula with a single frozen L is not an
exact description; conditioning and the dependence must be retained.

## Equal energy is not equal output risk

The included CPU example has 64 writes, scalar `A=0.99`, `q=1`, zero start,
and readout-before-storage. Every error is either +1 or -1, so every trajectory
has exactly one unit of squared error per write and 64 units in total. The
numbers below are exact finite-ensemble expectations evaluated in FP64, not
Monte Carlo means, measured codec errors, or model scores.

| Error ensemble | Mean at each time | Expected sum of readout squares | Cross-time covariance contribution |
|---|---:|---:|---:|
| Independent centered signs | 0 | 1,360.848513 | 0, up to FP64 roundoff |
| One random sign reused for all writes | 0 | 53,793.380821 | +52,432.532309 |
| One random sign, alternating at each write | 0 | 24.722763 | -1,336.125749 |
| Independent signs per eight-write block | 0 | 10,242.275819 | +8,881.427306 |

In particular, choosing one trajectory-wide random sign makes every marginal
mean zero but leaves covariance +1 between every pair of times. Centering is
not temporal independence. The IID cross contribution is stored as its raw
FP64 residual (about -2.27e-13), not clipped to fabricate exact zero.

There is a separate directional issue. On a permitted one-write state-error
subspace, `G=L^T L=omega I` is sufficient for `J=omega ||e||²`. The toy
isotropic map `L=2I` gives risk 4 in both unit coordinate directions; the map
`diag(1,3)` gives risks 1 and 9 at the same input energy. For repeated writes,
individual isotropic diagonal blocks do not eliminate the off-diagonal blocks
of the stacked response Gramian. Time-dependent weights and cross terms still
matter. A contraction gives a bound, not equality of risk at equal energy:
if `||A_t||<=rho<1`, then

`||delta_t|| <= sum_(j<=t) rho^(t-j) ||epsilon_j||`.

For the local scalar-decay, decay-before-overwrite GDN equation with unit keys
and `0<=beta<=1`, `A=alpha(I-beta k k^T)` is nonexpansive up to alpha in exact
arithmetic. Channel-wise GDN2 decay and gates require their actual nonsymmetric
operator. Neither fact says that every error direction is equally observable,
that errors are independent, or that adaptive quantization cannot fail.

## A limited stationary scalar calculation

Only under `|alpha|<1` and `epsilon_t=mu+eta_t` with independent centered
innovations of variance `sigma²` does the stored-state stationary second moment
take the familiar form

`E[delta_infinity²] = mu²/(1-alpha)² + sigma²/(1-alpha²)`.

For equal unit per-write input energy, a completely coherent sign sequence has
stationary second moment `1/(1-alpha)²`; independent centered signs give
`1/(1-alpha²)`. At alpha=0.99 their squared-error ratio is **199** and their RMS
ratio is **sqrt(199)=14.106736**. A trajectory-wide random sign realizes the
coherent value even though its ensemble mean is zero; it does **not** satisfy
the independent-innovation premise above. General bias/variance choices change
the ratio. These are scalar stored-state quantities, not output-KL predictions,
and do not explain the historical 16.34x model comparison.

## What DIAG top-k does and does not solve

Given fixed signed additive scores `s_i`, equal row cost, and exactly k protected
rows, maximizing `sum_(i selected) s_i` is solved by the k largest scores.
An exchange of a selected smaller score with an unselected larger score proves
the statement; ascending row index resolves ties reproducibly. Negative scores
are not clipped: exactly-k is different from at-most-k. A brute-force small
case checks the implementation.

This describes the DIAG **additive surrogate** with `s_i=K_ii+2c_i`, not the
full binary quadratic objective with off-diagonal K. It is not a theorem that
DIAG produces the best actual recurrence or Native-logit distribution. Continuous
multi-bit allocation and water-filling are different optimization problems and
are not newly implemented or claimed here.

For a physical injection map B, `M=B^T B` measures the declared input energy and
`K=B^T L^T L B` measures its response. K is not `L^T L` on arbitrary state errors.
Low-only error and low-minus-high injection differ; signed c and the high-tier
residual baseline cannot be dropped when interpreting row-promotion scores.

## Primary sources: what was actually checked

| Source | Verification in this cycle | Consequence for the claim |
|---|---|---|
| Mullis & Roberts (1976), [DOI](https://doi.org/10.1109/TCS.1976.1084254) | Publisher-deposited Crossref title/authors/date verified; IEEE body blocked by its access check | Historical state-space roundoff reference, **not** a newly verified proof or theorem attribution |
| Hwang (1977), [DOI](https://doi.org/10.1109/TASSP.1977.1162971) | Publisher-deposited bibliography verified; IEEE body not read | Same limitation; no original-body assumptions inferred from a search snippet |
| Kirac & Vaidyanathan (1996), [author-hosted paper](https://authors.library.caltech.edu/records/cnw88-69x52) | All 16 PDF pages text-read; printed pp. 816 and 821 visually checked | Subtractive independence requires input-independent Nyquist lattice dither; the process statement invokes IID time dithers; overflow is assumed avoided |
| Kozyrev & Maiboroda (2026), [arXiv 2609.04098v1](https://arxiv.org/html/2609.04098v1) | Primary HTML and mirrored text/captions read, including mechanism and limitation sections | W4A4 projection quantization and FP8 attention KV are not RTPA's persistent recurrent-state codec; no author-code execution or independent score reproduction |

The dither conditions do not automatically survive adaptive metadata, saturation,
finite-precision H32, or a reused random value. A subtractive implementation
would also need matching encoder/decoder randomness and a complete byte ledger.
This cycle's theoretical check does not add a dither policy or certify one.

The 2026 paper reports a bounded FP32 captured-input perturbation plateau and
impulse forgetting. Those observations can coexist with direction-dependent
finite-horizon risk and temporal correlation. Its weights/activations, model,
serving backend, and perturbation locations differ from this repository's
state writes. Four-bit weights and 9.4375-bit state payload are not directly
comparable bit budgets. The paper's measured plateau is not evidence that an
adaptive FP16 metadata representation cannot overflow.

## Model boundary and reproducible check

At fixed operands, the storage error identity is algebraic. In the actual model,
quantization can change later queries, keys, values and gates. Relative to a
reference transition, subtraction adds an operand-change forcing term, for
example `(A_method-A_ref) S_method_previous + (U_method-U_ref)`, besides local
storage/rounding error. Readout changes add further terms. Ignoring these terms
does not yield an exact end-to-end KL decomposition. The unresolved historical
BF16/FP32 reference discrepancy is not repaired by these synthetic identities.

Source, tests, seed and dimension/epsilon-based bounds were frozen before this
synthetic run. Direct recurrence versus dense response differed by 2.64e-16 in
Frobenius norm; the 16-trajectory covariance identity differed by 2.78e-17.
Six numerical identity checks and seven focused unit tests passed. These are
implementation checks, not confidence intervals or empirical model evidence.
The run uses no GPU, model, calibration capture, or network at execution time.

From a checkout with NumPy installed, choose a fresh output path:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
python -m rtpa_research.grid_theory \
  --preregistration data/benchmarks/grid_feedback/theory_sources.json \
  --out recomputed/grid-theory.json
PYTHONPATH=src python -m unittest discover -s tests -p test_grid_theory.py -v
```

The [frozen result](../results/grid_feedback/theory.json) stores formulas,
all synthetic operands, nonzero rounding residuals and the execution receipt.
The CLI rejects changed source/protocol and refuses to overwrite an existing
output. Standard algebra and classical noise analysis are not new RTPA
theorems; any contribution must come from reproducible conditions and actual
interventions on the stated recurrent-write implementation.
