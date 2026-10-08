| candidate | C1 coverage, pre (>= 0.90) | C2 median W/W_ref, pre (<= 1.25) | C3 median W/W_ref, control (<= 1.00) ; coverage control (>= 0.90) | C5 median relErr/relErr_ref, pre (<= 1.50) | median relErr, pre | median posterior p, pre | mean s per fit |
|---|---|---|---|---|---|---|---|
| base (power, L6-L10) | 0.883 (False) | 0.956 (True) | 0.234 ; 0.989 (True) | 2.161 (False) | 0.0347 | 0.76 | 37 |
| sat (saturating, L6-L10) | 0.889 (False) | 0.689 (True) | 0.234 ; 0.978 (True) | 1.150 (True) | 0.0107 | 1.25 | 52 |
| two (two-term, L6-L10) | 0.900 (True) | 3.170 (False) | 5.030 ; 0.883 (False) | 5.369 (False) | 0.0693 | 0.96 | 134 |
| ref (power, L8-L10) | 1.000 (True) | 1.000 (True) | 1.000 ; 1.000 (True) | 1.000 (True) | 0.0105 | 1.39 | 22 |

True p, median over the pre-asymptotic truths: 1.59.
C5 was added AFTER the round-1 results were seen (round 1: base at 2.67 on its own seeds); C1-C3 are the round-1 criteria applied to the candidate X in place of A and O replaced by ref.
Seeds: pre-asymptotic 200-219, control 300-319. Ref is a fit of its own on each truth.
Dropped fits (rhat > 1.05, re-run): 130; unconverged after 6 tries: 15.
sat, log h_s posterior: fraction of truths with 90% interval < half the prior log-range 0.95; corr(posterior median, true) over log h_s 0.64; posterior-median position in the prior interval (quartiles) [0.47, 0.49, 0.58].
