# AI-assisted plot generated from actual training histories.
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).parent
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
report = {}
for ax, agent, color in zip(axes, ['feedforward', 'recurrent'], ['#2563eb', '#b45309']):
    trajectories = []
    seed_stats = []
    for seed in range(3):
        path = root / 'results' / f'{agent}_seed{seed}_matches500_history.csv'
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 500
        assert [int(r['match']) for r in rows] == list(range(1, 501))
        scores = np.array([float(r['score']) for r in rows])
        rewards = np.array([float(r['reward']) for r in rows])
        losses = np.array([float(r['loss']) for r in rows])
        assert np.isin(scores, [0, 1, 2, 3]).all()
        assert np.allclose(rewards, scores / 3)
        assert np.isfinite(losses).all()
        rolling = np.convolve(scores, np.ones(50)/50, mode='valid')
        trajectories.append(rolling)
        seed_stats.append({'seed': seed, 'total_score': int(scores.sum()), 'mean_score': float(scores.mean()), 'first100_mean': float(scores[:100].mean()), 'last100_mean': float(scores[-100:].mean()), 'win_rate': float((scores == 3).mean())})
        ax.plot(np.arange(50, 501), rolling, color=color, alpha=.25, linewidth=1, label='Individual training seeds' if seed == 0 else None)
    data = np.stack(trajectories)
    ax.plot(np.arange(50, 501), data.mean(axis=0), color=color, linewidth=2.5, label='Mean across 3 training seeds')
    ax.set_title(agent.capitalize(), weight='bold')
    ax.set_xlabel('Training match')
    ax.set_ylim(0, 3)
    ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper left', fontsize=8)
    report[agent] = seed_stats
axes[0].set_ylabel('Mean match score (50-match rolling window)')
fig.suptitle('Restricted-observation actor–critic: training scores', weight='bold', fontsize=14)
fig.text(.5,.015,'Training outcomes, not frozen-policy evaluation. Thin lines show seed variation; no confidence interval is shown.',ha='center',fontsize=9,color='#475569')
fig.tight_layout(rect=[0,.045,1,.93])
fig.savefig(root / 'results' / 'learning_curves.png', dpi=180)
print(json.dumps(report, indent=2))
