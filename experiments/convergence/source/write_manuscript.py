"""Insert two measured vector figures into the original complete manuscript.

Text outside the visualization block is preserved exactly. The added numerical
claims are checked against the measured 4096-cycle source-data aggregates.
"""
from pathlib import Path
import argparse, difflib, json, re

ROOT = Path(__file__).resolve().parent.parent

INSERT = r'''% Measured finite-stream diagnostics: two separate two-panel figures.
\FloatBarrier
\paragraph{Formation of normalized stochastic attention.}
Using the fixed CIFAR-10 checkpoint (model seed 11) and its previously
calibrated operator, we sampled eight of 64 image/block/head attention
matrices from the first eight diagnostic test images without replacement
(selection seed 20261009). All 128 rows were retained, spanning five images.
Eight matched streams (seeds $101,202,\ldots,808$) each recorded 4096
categorical events after 256 burn-in cycles; shorter results are nested
prefixes. This diagnostic isolates attention normalization at fixed
numerical scores.

Let $\widehat a_j(L)$ be the fraction of the first $L$ recorded events
assigned to key $j\in\{1,\ldots,16\}$. One event per cycle ensures
non-negative frequencies with $\sum_{j=1}^{16}\widehat a_j(L)=1$ at every
non-empty prefix (Fig.~\ref{fig:frost_attention_formation}).
The representative row was selected at the upper-median reference-entropy
rank before sampling. Its three-group probability trajectory approaches
the stationary operator output while preserving total probability mass;
the stationary output can differ from exact softmax. All error metrics
below retain the original 16 components.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{Images/FROST_attention_formation.pdf}
\caption{\textbf{Formation of normalized attention from finite-state events.}
\textbf{a}, Actual trajectories of 16 input-driven finite-state machines
(FSMs), categorical output events and accumulated attention probabilities
for the representative score row, using stream seed 101. The first 128
recorded cycles are displayed in the state and event maps. Inputs and keys
are ordered by descending score; every event column selects exactly one
key. Finite probability columns are prefixes of the same stream, followed
by stationary tensor-train (TT) output and exact softmax.
\textbf{b}, A three-group projection of the 16-component probability
simplex: the top key, second key and sum of the other 14 keys. Grey paths
show eight FROST streams; colour highlights seed 101 as stream length
$L$ increases from 16 to 4096. The inset connects measured prefixes at
$L=1024$, 2048 and 4096. The star and diamond denote stationary TT and
exact softmax, respectively. Group masses are not conditionally
renormalized. $L$ excludes the 256 burn-in cycles.}
\label{fig:frost_attention_formation}
\end{figure}
\FloatBarrier

\paragraph{Finite-stream convergence and approximation error.}
FROST mean row total variation (TV) distance to exact softmax decreased
from $0.2267\pm0.0043$ at $L=64$ to $0.06509\pm0.00064$ at $L=4096$
(Fig.~\ref{fig:frost_attention_convergence}). The base TT control reached
$0.08151\pm0.00141$, giving FROST 20.1\% lower mean error at the same
16,512-parameter capacity. Stationary errors were 0.05509 for FROST and
0.07396 for base TT, whereas FROST error relative to its own stationary
output decreased from 0.2171 at $L=64$ to 0.02955 at $L=4096$.
These measured prefixes show decreasing sampling error; calibration
lowers the approximation floor. Uncertainties are sample standard
deviations of eight stream-level means at fixed weights and scores.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{Images/FROST_attention_convergence.pdf}
\caption{\textbf{Sampling convergence and the calibrated approximation floor.}
\textbf{a}, Row TV distance to exact softmax, averaged over eight streams
for each of 128 fixed test score rows, at stream lengths from 16 to 4096.
Both heatmaps use the same logarithmic colour scale and the same
reference-entropy row ordering.
\textbf{b}, Mean row TV versus $L$ for base TT and calibrated FROST;
shading denotes one sample standard deviation across eight stream-level
means. The dotted blue curve measures FROST error relative to its
stationary TT output. Dashed horizontal lines denote the two stationary
errors relative to exact softmax; the bracket compares measured mean
errors at $L=4096$. All operator parameters and Transformer weights are
fixed; both operators use 16 states, rank 8 and 16,512 parameters. These
are sampling-convergence measurements, with $L$ excluding the 256
burn-in cycles.}
\label{fig:frost_attention_convergence}
\label{fig:frost_convergence}
\end{figure}
\FloatBarrier

'''


def main(source):
    original = Path(source).read_text(encoding='utf-8')
    anchor = '% Requires graphicx and booktabs.\n\\paragraph{Image-specific coordinate reconstruction.}'
    assert original.count(anchor) == 1
    assert '\\label{fig:frost_attention_formation}' not in original
    updated = original.replace(anchor, INSERT+anchor, 1)
    (ROOT/'FROST_with_convergence.tex').write_text(updated, encoding='utf-8')
    (ROOT/'FROST_convergence_insert.tex').write_text(INSERT, encoding='utf-8')

    summary = json.loads((ROOT/'source_data/summary.json').read_text())
    def at(method, length):
        return next(row for row in summary[method]['series'] if row['L'] == length)
    first, final, base = at('FROST', 64), at('FROST', 4096), at('SC_base', 4096)
    claims = [
        (f"{first['mean_row_tv_mean']:.4f}", '0.2267'),
        (f"{first['mean_row_tv_stream_sd']:.4f}", '0.0043'),
        (f"{final['mean_row_tv_mean']:.5f}", '0.06509'),
        (f"{final['mean_row_tv_stream_sd']:.5f}", '0.00064'),
        (f"{base['mean_row_tv_mean']:.5f}", '0.08151'),
        (f"{base['mean_row_tv_stream_sd']:.5f}", '0.00141'),
        (f"{summary['paired_long_stream_gain_percent']:.1f}", '20.1'),
        (f"{summary['FROST']['analytic_mean_row_tv']:.5f}", '0.05509'),
        (f"{summary['SC_base']['analytic_mean_row_tv']:.5f}", '0.07396'),
        (f"{first['sampling_mean_row_tv_mean']:.4f}", '0.2171'),
        (f"{final['sampling_mean_row_tv_mean']:.5f}", '0.02955')]
    assert all(actual == expected for actual, expected in claims), claims
    assert updated.replace(INSERT, '', 1) == original
    assert not re.search(r'16[,{]?384|8192|8,192|16\{,\}384', INSERT)
    labels = re.findall(r'\\label\{([^}]+)\}', updated)
    refs = re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', updated)
    missing_refs = sorted(set(refs)-set(labels))
    assert not missing_refs, missing_refs
    bib = (ROOT/'FROST_references.bib').read_text()
    keys = set(re.findall(r'@\w+\s*\{\s*([^,\s]+)', bib))
    cited = set(key.strip() for group in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}', updated)
        for key in group.split(','))
    missing_cites = sorted(cited-keys)
    assert not missing_cites, missing_cites
    qa = dict(original_source_lines=len(original.splitlines()),
        updated_source_lines=len(updated.splitlines()), insertion_only=True,
        numeric_assertions_passed=len(claims), maximum_sampling_length=4096,
        all_references_resolve=True, missing_reference_labels=missing_refs,
        missing_bibliography_keys=missing_cites, citation_key_count=len(cited),
        new_figure_labels=['fig:frost_attention_formation', 'fig:frost_attention_convergence'],
        new_figure_paths=['Images/FROST_attention_formation.pdf', 'Images/FROST_attention_convergence.pdf'])
    (ROOT/'source_data/manuscript_checks.json').write_text(json.dumps(qa, indent=2)+'\n')
    (ROOT/'source_data/manuscript_changes.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile='uploaded_full_source.tex', tofile='FROST_with_convergence.tex')))
    print(json.dumps(qa, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    main(parser.parse_args().source)
