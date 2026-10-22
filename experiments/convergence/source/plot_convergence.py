"""Two spacious publication figures from actual measured 4096-cycle prefixes.

Heatmaps, colour bars, trajectories and text are vector objects in PDF/SVG.
No smoothing, invented trajectories or extended sampling horizon is used.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, colors, ticker
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parent.parent
for name in ['NimbusSans-Regular.ttf', 'NimbusSans-Bold.ttf']:
    fm.fontManager.addfont(str(ROOT / 'fonts' / name))
plt.rcParams.update({
    'font.family': 'Nimbus Sans', 'font.size': 8.5,
    'axes.labelsize': 8.7, 'axes.titlesize': 9.0,
    'xtick.labelsize': 8.2, 'ytick.labelsize': 8.2,
    'axes.linewidth': .7, 'xtick.major.width': .7, 'ytick.major.width': .7,
    'xtick.major.size': 2.8, 'ytick.major.size': 2.8,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.fontset': 'stixsans',
    'text.color': '#181818', 'axes.labelcolor': '#181818',
    'xtick.color': '#181818', 'ytick.color': '#181818',
    'savefig.facecolor': 'white'})
TEAL = '#0B8C86'
BLUE = '#08739E'
ORANGE = '#BC650D'
INK = '#181818'
GREY = '#777777'
STATE = colors.LinearSegmentedColormap.from_list('state',
    ['#F5FAFA', '#C5E0DD', '#65B3AC', '#0B8C86', '#084B4A'])
ERROR = colors.LinearSegmentedColormap.from_list('TV',
    ['#F5FAFD', '#CCE4F1', '#83BBD0', '#2A82A4', '#153F5B'])
PROB = colors.LinearSegmentedColormap.from_list('prob',
    ['#F6FAFD', '#C4E0EB', '#519BB6', '#085B7B'])
TIME = colors.LinearSegmentedColormap.from_list('stream_time',
    ['#CCDCE5', '#08739E', '#0B8C86'])


def panel(fig, x, y, letter, title):
    fig.text(x, y, letter, fontsize=12, fontweight='bold', va='baseline')
    fig.text(x+.035, y, title, fontsize=9.5, fontweight='bold', va='baseline')


def mesh(ax, data, cmap, norm=None, vmin=None, vmax=None):
    ny, nx = data.shape
    result = ax.pcolormesh(np.arange(nx+1)-.5, np.arange(ny+1)-.5, data,
        cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, shading='flat',
        rasterized=False, edgecolors='none', linewidth=0, antialiased=False)
    ax.set_ylim(ny-.5, -.5)
    ax.set_xlim(-.5, nx-.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=3)
    return result


def colorbar(fig, artist, bounds, ticks, bars, orientation='vertical'):
    cb = fig.colorbar(artist, cax=fig.add_axes(bounds), ticks=ticks,
        orientation=orientation)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, pad=3, labelsize=8.1)
    cb.ax.xaxis.set_minor_locator(ticker.NullLocator())
    cb.ax.yaxis.set_minor_locator(ticker.NullLocator())
    bars.append(cb)
    return cb


def simplex_xy(p):
    return np.column_stack([p[:, 1]+p[:, 2]/2, p[:, 2]*np.sqrt(3)/2])


def save(fig, stem, bars):
    # Pin the supplied TrueType font, avoiding an installed CFF font with
    # the same family name. Draw once to materialize all tick labels.
    fig.canvas.draw()
    for cb in bars:
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor('face')
    for txt in fig.findobj(Text):
        weight = txt.get_fontweight()
        bold = weight in ['bold', 'semibold', 'heavy', 'black'] or (
            isinstance(weight, (int, float)) and weight >= 600)
        font = ROOT/'fonts'/('NimbusSans-Bold.ttf' if bold else 'NimbusSans-Regular.ttf')
        txt.set_fontproperties(fm.FontProperties(fname=str(font), family='Nimbus Sans',
            weight='bold' if bold else 'normal', size=txt.get_fontsize()))
    fig.savefig(ROOT/f'{stem}.pdf')
    fig.savefig(ROOT/f'{stem}.svg')
    fig.savefig(ROOT/f'{stem}.png', dpi=360)
    plt.close(fig)
    print('Saved', stem, flush=True)


def formation(inp, lengths, traces):
    fig = plt.figure(figsize=(180/25.4, 115/25.4), facecolor='white')
    bars = []
    rep = int(inp['representative_row'])
    reference = inp['reference']
    t = traces['FROST'][0]
    order = t['order'][rep]

    # a: actual finite states, emissions and probability accumulation.
    panel(fig, .026, .956, 'a', 'Finite states to attention')
    left, width = .091, .413
    ax = fig.add_axes([left, .710, width, .184])
    artist = mesh(ax, t['representative_states'][:128].T, STATE, vmin=0, vmax=15)
    ax.set_yticks([0, 7, 15], ['1', '8', '16'])
    ax.set_xticks([])
    ax.set_ylabel('Score rank', labelpad=6)
    ax.set_title('Input-driven FSM states', loc='left', pad=5)
    cb = colorbar(fig, artist, [.519, .710, .012, .184], [0, 7, 15], bars)
    cb.set_label('State', labelpad=4, fontsize=8.3)

    ax = fig.add_axes([left, .505, width, .114])
    events = t['events_sorted'][:128, rep]
    onehot = (np.arange(16)[:, None] == events[None]).astype(np.uint8)
    assert np.all(onehot.sum(0) == 1)
    mesh(ax, onehot, colors.ListedColormap(['#F3F7F9', BLUE]), vmin=0, vmax=1)
    ax.set_yticks([0, 7, 15], ['1', '8', '16'])
    ax.set_ylabel('Key rank', labelpad=6)
    ax.set_xticks([0, 63, 127], ['1', '64', '128'])
    ax.set_xlabel('Recorded cycle, $t$', labelpad=4)
    ax.set_title('One categorical key per cycle', loc='left', pad=5)

    ax = fig.add_axes([left, .170, width, .205])
    shown = [16, 64, 256, 1024, 4096]
    columns = [t['probabilities'][list(lengths).index(L), rep, order] for L in shown]
    columns += [t['analytic'][rep, order], reference[rep, order]]
    table = np.stack(columns, axis=1)
    upper = .05*np.ceil(table.max()/.05)
    artist = mesh(ax, table, PROB, vmin=0, vmax=upper)
    ax.set_yticks([0, 7, 15], ['1', '8', '16'])
    ax.set_ylabel('Key rank', labelpad=6)
    ax.set_xticks(np.arange(7), ['16', '64', '256', '1,024', '4,096', 'TT\nlimit', 'Exact\nsoftmax'])
    ax.tick_params(axis='x', labelsize=8.0, pad=4)
    ax.set_title('Accumulated attention probabilities', loc='left', pad=5, fontsize=8.8)
    for edge in np.arange(.5, 6):
        ax.axvline(edge, color='white', lw=.6)
    cb = colorbar(fig, artist, [.519, .170, .012, .205], [0, upper/2, upper], bars)
    cb.ax.set_yticklabels([f'{x:g}' for x in [0, upper/2, upper]])
    cb.ax.set_title('Probability', fontsize=8.0, pad=6)
    fig.text(left, .083, 'Finite columns: stream length, $L$\nRepresentative row; stream seed 101',
        fontsize=8.1, color=GREY, linespacing=1.55, va='top')

    # b: a mass-preserving three-group simplex, not a conditional projection.
    panel(fig, .593, .956, 'b', 'Normalized prefix trajectories')
    fig.text(.608, .916, r'$\sum_{j=1}^{16}\widehat a_j(L)=1$ at every prefix',
        fontsize=9.1, color=TEAL)
    ax = fig.add_axes([.591, .405, .367, .471])
    ax.set_aspect('equal')
    triangle = np.array([[0, 0], [1, 0], [.5, np.sqrt(3)/2], [0, 0]])
    ax.plot(triangle[:, 0], triangle[:, 1], color='#AEB8BE', lw=.8)
    for q in [.2, .4, .6, .8]:
        segments = [np.array([[q, 0], [q/2, q*np.sqrt(3)/2]]),
            np.array([[1-q, 0], [1-q/2, q*np.sqrt(3)/2]]),
            np.array([[q/2, q*np.sqrt(3)/2], [1-q/2, q*np.sqrt(3)/2]])]
        for segment in segments:
            ax.plot(segment[:, 0], segment[:, 1], color='#E6EDF0', lw=.45, zorder=0)
    ax.text(-.01, -.061, 'Top key', ha='left', fontsize=8.3)
    ax.text(1.01, -.061, 'Second key', ha='right', fontsize=8.3)
    ax.text(.5, .923, 'Other 14 keys', ha='center', fontsize=8.3)
    ax.set_xlim(-.07, 1.07)
    ax.set_ylim(-.11, .966)
    ax.axis('off')
    sampled = np.unique(np.r_[np.arange(16, 65),
        np.geomspace(64, 4096, 360).round().astype(int)])
    time_norm = colors.LogNorm(vmin=16, vmax=4096)
    paths = []
    for trace in traces['FROST']:
        ev = trace['events_sorted'][:, rep]
        assert len(ev) == 4096
        groups = np.stack([np.cumsum(ev == 0), np.cumsum(ev == 1),
            np.cumsum(ev >= 2)], axis=1)
        p = groups/np.arange(1, len(ev)+1)[:, None]
        assert np.allclose(p.sum(-1), 1, rtol=0, atol=3e-16)
        xy = simplex_xy(p)
        paths.append(xy)
        ax.plot(xy[sampled-1, 0], xy[sampled-1, 1], color='#809B9D',
            lw=.65, alpha=.52, zorder=2)
    xy = paths[0][sampled-1]
    line = LineCollection(np.stack([xy[:-1], xy[1:]], axis=1),
        cmap=TIME, norm=time_norm, linewidth=1.3, zorder=4)
    line.set_array(sampled[:-1])
    ax.add_collection(line)
    for L in [16, 64, 256, 1024, 4096]:
        ax.scatter(*paths[0][L-1], s=17, color=TIME(time_norm(L)),
            edgecolor='white', lw=.4, zorder=5)

    def grouped(a):
        return np.array([[a[order[0]], a[order[1]], a[order[2:]].sum()]])

    limit_xy = simplex_xy(grouped(t['analytic'][rep]))[0]
    exact_xy = simplex_xy(grouped(reference[rep]))[0]
    ax.scatter(*limit_xy, marker='*', s=77, facecolor=TEAL, edgecolor='white', lw=.5, zorder=7)
    ax.scatter(*exact_xy, marker='D', s=31, facecolor='white', edgecolor=INK, lw=.85, zorder=8)

    zoom = fig.add_axes([.620, .167, .319, .197], facecolor='white')
    terminal = np.array([1024, 2048, 4096])-1
    for si, xy in enumerate(paths):
        zoom.plot(xy[terminal, 0], xy[terminal, 1], color=TEAL,
            alpha=.88 if si == 0 else .48, lw=1 if si == 0 else .6,
            marker='o', markersize=2.2)
    ends = np.stack([xy[-1] for xy in paths])
    points = np.vstack([*[xy[terminal] for xy in paths], limit_xy, exact_xy])
    lo, hi = points.min(0), points.max(0)
    padding = np.maximum((hi-lo)*.23, .0025)
    zoom.set_xlim(lo[0]-padding[0], hi[0]+padding[0])
    zoom.set_ylim(lo[1]-padding[1], hi[1]+padding[1])
    zoom.scatter(ends[:, 0], ends[:, 1], s=13, c=TEAL, edgecolors='white', lw=.35, zorder=5)
    zoom.scatter(*limit_xy, marker='*', s=75, facecolor=TEAL, edgecolor='white', lw=.5, zorder=7)
    zoom.scatter(*exact_xy, marker='D', s=30, facecolor='white', edgecolor=INK, lw=.85, zorder=8)
    zoom.set_xticks([])
    zoom.set_yticks([])
    for spine in zoom.spines.values():
        spine.set_color('#AEB8BE')
        spine.set_linewidth(.6)
    zoom.text(.03, .96, r'Zoom: $L=1,024\ \rightarrow\ 4,096$', transform=zoom.transAxes,
        fontsize=8.1, va='top', bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=.8))
    ax.indicate_inset_zoom(zoom, edgecolor='#8A969A', alpha=.7, linewidth=.55)
    handles = [Line2D([], [], marker='*', mec='white', mfc=TEAL, ls='', ms=8, label='Stationary TT'),
        Line2D([], [], marker='D', mec=INK, mfc='white', ls='', ms=4.7, label='Exact softmax')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.603, .113),
        ncol=2, frameon=False, fontsize=8.1, handletextpad=.35, columnspacing=.9, borderaxespad=0)
    cb = colorbar(fig, plt.cm.ScalarMappable(norm=time_norm, cmap=TIME),
        [.639, .043, .283, .010], [16, 256, 4096], bars, orientation='horizontal')
    cb.ax.set_xticklabels(['16', '256', '4,096'])
    fig.text(.613, .041, '$L$', fontsize=8.5, va='bottom')
    save(fig, 'FROST_attention_formation', bars)


def convergence(inp, lengths, traces, summary):
    fig = plt.figure(figsize=(180/25.4, 100/25.4), facecolor='white')
    bars = []
    panel(fig, .026, .947, 'a', 'Convergence across score rows')
    fig.text(.079, .890, 'Row-wise error averaged over eight streams', fontsize=8.3, color=GREY)
    atlas = {method: np.mean([t['row_tv'] for t in traces[method]], axis=0).T[
        inp['entropy_order']] for method in traces}
    norm = colors.LogNorm(vmin=.01, vmax=.8)
    values = np.concatenate([v.ravel() for v in atlas.values()])
    assert values.min() >= .01 and values.max() <= .8
    for i, method in enumerate(['SC_base', 'FROST']):
        ax = fig.add_axes([.079+i*.231, .235, .204, .580])
        artist = mesh(ax, atlas[method], ERROR, norm=norm)
        ax.set_yticks([0, 31, 63, 95, 127], ['1', '32', '64', '96', '128'] if i == 0 else ['']*5)
        if i == 0:
            ax.set_ylabel('Row rank by reference entropy', labelpad=6)
        ax.set_xticks([0, 2, 4, 6, 8], ['16', '64', '256', '1,024', '4,096'])
        ax.tick_params(axis='x', rotation=40, labelsize=8.0, pad=4)
        ax.set_xlabel('Stream length, $L$', labelpad=3)
        ax.set_title('Base TT' if i == 0 else 'Calibrated FROST',
            color=ORANGE if i == 0 else TEAL, fontsize=8.8, fontweight='bold', pad=7)
    cb = colorbar(fig, artist, [.528, .235, .012, .580], [.01, .02, .05, .1, .2, .5, .8], bars)
    cb.ax.set_yticklabels(['0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '0.8'])
    cb.ax.set_title('TV', fontsize=8.3, pad=7)

    panel(fig, .618, .947, 'b', 'Sampling and approximation')
    gain = summary['paired_long_stream_gain_percent']
    fig.text(.654, .888, f'$L=4,096$: {gain:.1f}% lower TV', fontsize=9.0, color=TEAL)
    ax = fig.add_axes([.673, .235, .302, .579])
    ax.spines[['top', 'right']].set_visible(False)
    for method, color in [('SC_base', ORANGE), ('FROST', TEAL)]:
        series = summary[method]['series']
        mu = np.array([r['mean_row_tv_mean'] for r in series])
        sd = np.array([r['mean_row_tv_stream_sd'] for r in series])
        ax.fill_between(lengths, mu-sd, mu+sd, color=color, alpha=.18, lw=0)
        ax.plot(lengths, mu, color=color, lw=1.45, marker='o', markersize=3.4,
            markeredgewidth=.45, markeredgecolor='white',
            label='Base TT to softmax' if method == 'SC_base' else 'FROST to softmax')
        ax.axhline(summary[method]['analytic_mean_row_tv'], color=color,
            ls=(0, (4, 2.6)), lw=.9, alpha=.9)
    series = summary['FROST']['series']
    mu = np.array([r['sampling_mean_row_tv_mean'] for r in series])
    sd = np.array([r['sampling_mean_row_tv_stream_sd'] for r in series])
    ax.fill_between(lengths, mu-sd, mu+sd, color=BLUE, alpha=.13, lw=0)
    ax.plot(lengths, mu, color=BLUE, ls=(0, (1.3, 1.8)), lw=1.15,
        marker='s', markersize=2.8, markeredgewidth=.45, markeredgecolor='white',
        label='FROST to stationary TT')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xlim(12, 5900)
    ax.set_ylim(.022, .57)
    ax.set_xticks([16, 64, 256, 1024, 4096], ['16', '64', '256', '1,024', '4,096'])
    ax.tick_params(axis='x', rotation=40, labelsize=8.0, pad=4)
    ax.set_yticks([.032, .064, .128, .256, .512], ['0.032', '0.064', '0.128', '0.256', '0.512'])
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel('Stream length, $L$', labelpad=3)
    ax.set_ylabel('Mean row TV', labelpad=6)
    ax.grid(axis='y', color='#E8EEF1', lw=.45, zorder=0)
    ax.legend(loc='upper right', frameon=False, fontsize=7.9, borderaxespad=.15,
        labelspacing=.45, handlelength=1.65, handletextpad=.4)
    base = summary['SC_base']['series'][-1]['mean_row_tv_mean']
    frost = summary['FROST']['series'][-1]['mean_row_tv_mean']
    ax.plot([4096, 4750, 4750, 4096], [base, base, frost, frost], color=TEAL, lw=.85)
    ax.text(24, .080, 'Base TT limit', fontsize=8.0, color=ORANGE)
    ax.text(24, .045, 'FROST limit', fontsize=8.0, color=TEAL)
    fig.text(.079, .072, 'Both atlases share row order and colour scale.', fontsize=8.2, color=GREY)
    fig.text(.654, .072, 'Shading: stream mean ± one s.d.', fontsize=8.2, color=GREY)
    fig.text(.5, .018, '128 fixed score rows  |  16 states  |  TT rank 8  |  8 matched stream seeds',
        ha='center', fontsize=8.2, color=GREY)
    save(fig, 'FROST_attention_convergence', bars)


def main():
    with np.load(ROOT/'source_data/inputs.npz') as saved:
        inp = {key: saved[key] for key in saved.files}
    protocol = json.loads((ROOT/'source_data/protocol.json').read_text())
    summary = json.loads((ROOT/'source_data/summary.json').read_text())
    lengths = np.array(protocol['lengths'])
    assert lengths.max() == 4096 and len(lengths) == 9
    traces = {}
    for method in ['SC_base', 'FROST']:
        traces[method] = []
        for seed in protocol['stream_seeds']:
            with np.load(ROOT/f'source_data/trace_{method}_seed{seed}.npz') as saved:
                traces[method].append({key: saved[key] for key in saved.files})
    formation(inp, lengths, traces)
    convergence(inp, lengths, traces, summary)


if __name__ == '__main__':
    main()
