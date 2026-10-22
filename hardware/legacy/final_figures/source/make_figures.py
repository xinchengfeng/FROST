"""Publication vector figures from unchanged verified L64 measurements."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path(__file__).resolve().parents[1]
D = json.loads((R / 'source/L64_summary.json').read_text())
B = {x['design']: x for x in D['baseline']}
F = next(x for x in D['variants'] if x['design'] == 'l64c')
items = [B['fp32_1'], B['fp32_4'], B['fx16'], F]
labels = ['FP32 · 1 lane', 'FP32 · 4 lanes', 'FX16 · 4 lanes', 'FROST']
colors = ['#9BA9C9', '#40577D', '#6AB3BE', '#D65348']
ink, gray, accent = '#263044', '#687181', '#BA4239'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.linewidth': .6,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'text.color': ink,
    'axes.labelcolor': ink, 'xtick.color': gray, 'ytick.color': gray,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.facecolor': 'white'})

def save(fig, name):
    for ext in ['pdf', 'svg']:
        fig.savefig(R / 'Images' / f'{name}.{ext}')
    fig.savefig(R / f'{name}.png', dpi=220)
    plt.close(fig)

def clean(ax, grid='x'):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#B7BECA')
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=3, width=.6)
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color='#EBEDF2', linewidth=.55)

# Figure 1: resource classes with substantial savings; complete FF data remain in the table.
fig = plt.figure(figsize=(7.2, 3.20))
specs = [(.205, .292, 'comb', 'Combinational logic', 30200, [0, 10000, 20000], ['0', '10k', '20k']),
         (.56, .175, 'dsp', 'DSP blocks', 20.4, [0, 8, 16], ['0', '8', '16']),
         (.79, .175, 'bram18', 'Block RAMs', 3.9, [0, 1, 2, 3], ['0', '1', '2', '3'])]
for k, (x, w, key, title, maximum, ticks, ticklabels) in enumerate(specs):
    fig.text(x-.029, .926, chr(97+k), fontsize=11, fontweight='bold', va='center')
    fig.text(x, .926, title, fontsize=9, va='center')
    headline = '73.3% fewer COMB' if k == 0 else '0 DSP blocks' if k == 1 else '0 block RAMs'
    fig.text(x, .824, headline, fontsize=10.2, fontweight='bold', color=accent)
    fig.text(x, .765, 'vs FP32 · 4 lanes' if k == 0 else 'FROST · L = 64', fontsize=7.4, color=gray)
    ax = fig.add_axes([x, .208, w, .493])
    ax.axhspan(2.59, 3.41, color='#FDF0EC', zorder=0)
    for y, (row, col) in enumerate(zip(items, colors)):
        val = row[key]
        ax.barh(y, val, height=.48, color=col, zorder=3)
        if val == 0:
            ax.plot(0, y, 'o', color=accent, ms=4.8, clip_on=False, zorder=4)
        ax.text(val+maximum*(.060 if val == 0 else .035), y, f'{val:,}', va='center', fontsize=8.2,
                color=accent if y == 3 else ink, fontweight='bold' if y == 3 else 'normal')
    ax.set(xlim=(0, maximum), ylim=(3.6, -.62), yticks=np.arange(4))
    ax.set_yticklabels(labels if k == 0 else [])
    ax.tick_params(axis='y', pad=8, labelsize=8.2)
    if k == 0:
        ax.get_yticklabels()[-1].set_color(accent)
        ax.get_yticklabels()[-1].set_fontweight('bold')
    ax.set_xticks(ticks, ticklabels)
    ax.set_xlabel('Resource count', fontsize=7.4, labelpad=7)
    clean(ax)
fig.text(.205, .043, 'Post-route FPGA · ECP5 LFE5U-85F · support logic and storage included',
         fontsize=7.1, color=gray)
save(fig, 'FROST_hardware_stage3')

# Figure 2: absolute energy and area-power position, with all three arithmetic controls.
fig = plt.figure(figsize=(7.2, 3.75))
fig.text(.025, .937, 'a', fontsize=11, fontweight='bold')
fig.text(.065, .937, 'Energy per attention tile', fontsize=9)
fig.text(.607, .937, 'b', fontsize=11, fontweight='bold')
fig.text(.65, .937, 'Area and active power', fontsize=9)

ax = fig.add_axes([.176, .255, .284, .553])
ax.axhspan(2.59, 3.41, color='#FDF0EC', zorder=0)
for y, (row, col) in enumerate(zip(items, colors)):
    val = row['energy_uJ']
    ax.barh(y, val, height=.47, color=col, zorder=3)
    if y == 3:
        ax.errorbar(val, y, xerr=row['energy_uJ_sd'], fmt='none', ecolor=ink,
                    capsize=2.5, linewidth=.8, zorder=4)
    ax.text(val+.035, y, f'{val:.4f}', va='center', fontsize=8,
            fontweight='bold' if y == 3 else 'normal', color=accent if y == 3 else ink)
ax.set(xlim=(0, 1.90), ylim=(3.6, -.62), yticks=np.arange(4), yticklabels=labels)
ax.set_xticks([0, .5, 1, 1.5])
ax.set_xlabel('Energy per tile (µJ)', labelpad=7)
ax.tick_params(axis='y', pad=7, labelsize=8)
ax.get_yticklabels()[-1].set_color(accent)
ax.get_yticklabels()[-1].set_fontweight('bold')
clean(ax)
fig.text(.528, .838, 'FROST\nsaving', fontsize=7, color=gray, ha='center', linespacing=1.25)
for y, row in enumerate(items[:3]):
    reduction = 100*(1-F['energy_uJ']/row['energy_uJ'])
    yf = fig.transFigure.inverted().transform(ax.transData.transform((0, y)))[1]
    fig.text(.528, yf, f'{reduction:.1f}%', fontsize=9, fontweight='bold', color=accent,
             ha='center', va='center')

ax2 = fig.add_axes([.642, .255, .322, .553])
ax2.set(xlim=(50, 95), ylim=(2.2, 4.05), xticks=[50, 60, 70, 80, 90], yticks=[2.5, 3, 3.5, 4])
ax2.grid(color='#EBEDF2', linewidth=.55)
ax2.set_axisbelow(True)
ax2.spines[['top', 'right']].set_visible(False)
for spine in ax2.spines.values(): spine.set_color('#B7BECA')
ax2.tick_params(length=3, width=.6)
ax2.set_xlabel('Mapped area (10³ µm²)', labelpad=7)
ax2.set_ylabel('Active power (mW)', labelpad=6)
markers = ['o', 's', '^', 'D']
for row, col, marker in zip(items, colors, markers):
    x, y = row['area_um2']/1000, row['power_mW']
    ax2.scatter(x, y, s=39 if row is not F else 59, c=col, marker=marker,
                edgecolor='white', linewidth=.8, zorder=5)
    if row is F:
        ax2.errorbar(x, y, yerr=row['power_mW_sd'], fmt='none', ecolor=ink,
                     capsize=2, linewidth=.7, zorder=4)
annotations = [
    (items[0], 'FP32 · 1 lane', (60.5, 3.87), colors[0]),
    (items[1], 'FP32 · 4 lanes', (80, 3.87), colors[1]),
    (items[2], 'FX16 · 4 lanes', (68.5, 2.90), colors[2]),
    (F, 'FROST', (65.6, 2.42), accent)]
for row, label, pos, col in annotations:
    ax2.annotate(label, xy=(row['area_um2']/1000, row['power_mW']), xytext=pos,
                 fontsize=7.8, color=accent if row is F else ink,
                 fontweight='bold' if row is F else 'normal',
                 ha='center', va='center', arrowprops={'arrowstyle':'-', 'color':col, 'lw':.6}, zorder=6)
ax2.annotate('Lower area\nand power', xy=(73, 2.45), xytext=(87.5, 2.93),
             fontsize=7, color=gray, ha='center', va='center',
             arrowprops={'arrowstyle':'->','color':'#9AA9AF','lw':.8})
fig.text(.51, .12, 'FROST: L = 64 · 16 queries per tile', ha='center', fontsize=7.4, color=ink)
fig.text(.51, .055, 'Nangate45 · 1.10 V · 25 °C · 15 MHz · pre-layout estimates',
         ha='center', fontsize=7.1, color=gray)
save(fig, 'FROST_hardware_tradeoffs')
print('Saved two PDF/SVG figures from unchanged verified data.')
