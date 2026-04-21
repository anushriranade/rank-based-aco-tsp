import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import random

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Rank-Based ACO | TSP", layout="wide", page_icon="🐜")

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0e1a;
    color: #e8eaf6;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1530 50%, #0a1628 100%);
}

h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

.metric-card {
    background: linear-gradient(145deg, #111827, #1a2540);
    border: 1px solid #2a3a5c;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.metric-card .label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    color: #7986cb;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-card .value {
    font-size: 32px;
    font-weight: 800;
    color: #a5d8ff;
}
.metric-card .sub {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #546e7a;
    margin-top: 4px;
}

.route-box {
    background: linear-gradient(135deg, #1a2540, #111827);
    border: 1px solid #3949ab;
    border-radius: 12px;
    padding: 24px;
    font-family: 'Space Mono', monospace;
    font-size: 15px;
    color: #80cbc4;
    text-align: center;
    letter-spacing: 3px;
    box-shadow: inset 0 0 40px rgba(57,73,171,0.1), 0 8px 32px rgba(0,0,0,0.5);
}

.iter-row {
    background: rgba(26, 37, 64, 0.6);
    border-left: 3px solid #3949ab;
    border-radius: 6px;
    padding: 8px 14px;
    margin-bottom: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    display: flex;
    justify-content: space-between;
}

.best-row {
    border-left-color: #00bcd4 !important;
    background: rgba(0, 188, 212, 0.08) !important;
    color: #00bcd4;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    color: #3949ab;
    text-transform: uppercase;
    border-bottom: 1px solid #1a2540;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

.rank-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1a237e, #283593);
    border: 1px solid #3949ab;
    border-radius: 20px;
    padding: 4px 14px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #7986cb;
    margin: 4px;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1530 0%, #0a0e1a 100%);
    border-right: 1px solid #1a2540;
}

.stSlider > div { color: #7986cb; }

button[kind="primary"] {
    background: linear-gradient(135deg, #1a237e, #3949ab) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 1px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  RANK-BASED ACO ALGORITHM
# ─────────────────────────────────────────────
def rank_based_aco(D, n_ants=10, n_iter=50, alpha=1.0, beta=2.0, rho=0.1, w=3, Q=100):
    N = len(D)
    tau = np.ones((N, N), dtype=float)
    eta = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j:
                eta[i][j] = 1.0 / D[i][j] if D[i][j] > 0 else 1e6

    best_tour = None
    best_cost = float('inf')
    history = []

    for iteration in range(n_iter):
        all_tours = []
        all_costs = []

        for _ in range(n_ants):
            visited = [False] * N
            start = random.randint(0, N - 1)
            tour = [start]
            visited[start] = True

            for _ in range(N - 1):
                current = tour[-1]
                probs = []
                candidates = []
                for j in range(N):
                    if not visited[j]:
                        p = (tau[current][j] ** alpha) * (eta[current][j] ** beta)
                        probs.append(p)
                        candidates.append(j)
                total = sum(probs)
                if total == 0:
                    next_city = random.choice(candidates)
                else:
                    probs = [p / total for p in probs]
                    next_city = random.choices(candidates, weights=probs)[0]
                tour.append(next_city)
                visited[next_city] = True

            cost = sum(D[tour[i]][tour[(i + 1) % N]] for i in range(N))
            all_tours.append(tour)
            all_costs.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_tour = tour[:]

        # Rank ants by cost (ascending = better)
        ranked = sorted(zip(all_costs, all_tours), key=lambda x: x[0])

        # Evaporate
        tau *= (1 - rho)

        # Deposit: only top-w ants + best-so-far contribute
        for rank_idx, (cost, tour) in enumerate(ranked[:w]):
            weight = w - rank_idx  # rank 1 → weight w, rank w → weight 1
            deposit = weight * Q / cost
            for i in range(N):
                a, b = tour[i], tour[(i + 1) % N]
                tau[a][b] += deposit
                tau[b][a] += deposit

        # Best-so-far also deposits (elite)
        if best_tour is not None:
            deposit = w * Q / best_cost
            for i in range(N):
                a, b = best_tour[i], best_tour[(i + 1) % N]
                tau[a][b] += deposit
                tau[b][a] += deposit

        history.append({
            'iteration': iteration + 1,
            'best_cost': best_cost,
            'iter_best': ranked[0][0],
            'best_tour': best_tour[:]
        })

    return best_tour, best_cost, history, tau


# ─────────────────────────────────────────────
#  DISTANCE MATRIX
# ─────────────────────────────────────────────
D_DEFAULT = np.array([
    [0,  10, 12, 11, 14],
    [10,  0, 13, 15,  8],
    [12, 13,  0,  9, 14],
    [11, 15,  9,  0, 16],
    [14,  8, 14, 16,  0],
], dtype=float)

N = 5
CITIES = [f"C{i}" for i in range(N)]

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    st.markdown("---")
    n_ants  = st.slider("Number of Ants",        5,  50, 10)
    n_iter  = st.slider("Iterations",           10, 200, 50)
    alpha   = st.slider("Alpha (τ weight)",    0.5, 3.0, 1.0, 0.1)
    beta    = st.slider("Beta (η weight)",     0.5, 5.0, 2.0, 0.1)
    rho     = st.slider("Evaporation (ρ)",    0.01, 0.5, 0.1, 0.01)
    w       = st.slider("Rank window (w)",       1,   5,   3)
    Q       = st.slider("Pheromone Q",          10, 500, 100, 10)

    st.markdown("---")
    run_btn = st.button("🚀 RUN RANK-BASED ACO", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:2.4rem; margin-bottom:0;'>
    🐜 Rank-Based Ant Colony Optimization
</h1>
<p style='color:#546e7a; font-family:Space Mono,monospace; font-size:13px; margin-top:4px;'>
    TRAVELLING SALESMAN PROBLEM · 5 CITIES · RAS VARIANT
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Distance Matrix Display
col_m, col_s = st.columns([1.2, 1])
with col_m:
    st.markdown('<div class="section-header">📐 Distance Matrix</div>', unsafe_allow_html=True)
    fig_mat, ax = plt.subplots(figsize=(4.5, 3.2))
    fig_mat.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    im = ax.imshow(D_DEFAULT, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(CITIES, color='#a5d8ff', fontsize=11)
    ax.set_yticklabels(CITIES, color='#a5d8ff', fontsize=11)
    for i in range(N):
        for j in range(N):
            ax.text(j, i, int(D_DEFAULT[i][j]), ha='center', va='center',
                    color='white' if D_DEFAULT[i][j] > 8 else '#333', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#546e7a')
    for spine in ax.spines.values(): spine.set_edgecolor('#1a2540')
    fig_mat.colorbar(im, ax=ax).ax.yaxis.set_tick_params(color='#546e7a')
    plt.tight_layout()
    st.pyplot(fig_mat)
    plt.close()

with col_s:
    st.markdown('<div class="section-header">ℹ️ How RAS Works</div>', unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:13px; color:#90a4ae; line-height:1.9;'>
<b style='color:#7986cb;'>1. Construct</b> — Each ant builds a tour using pheromone (τ) and heuristic (η) info.<br>
<b style='color:#7986cb;'>2. Rank</b> — Ants sorted by tour cost (best first).<br>
<b style='color:#7986cb;'>3. Deposit</b> — Only top <b style='color:#a5d8ff;'>w</b> ants deposit pheromone, weighted by rank:<br>
&nbsp;&nbsp;&nbsp;Rank 1 → weight <b>w</b>, Rank 2 → weight <b>w-1</b> … Rank w → weight <b>1</b><br>
<b style='color:#7986cb;'>4. Elite</b> — Best-so-far ant also deposits with weight <b>w</b>.<br>
<b style='color:#7986cb;'>5. Evaporate</b> — τ decays by factor (1 − ρ) each iteration.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
#  RUN / RESULTS
# ─────────────────────────────────────────────
if run_btn or 'aco_result' not in st.session_state:
    with st.spinner("🐜 Ants exploring paths..."):
        time.sleep(0.3)
        best_tour, best_cost, history, final_tau = rank_based_aco(
            D_DEFAULT, n_ants, n_iter, alpha, beta, rho, w, Q
        )
        st.session_state['aco_result'] = (best_tour, best_cost, history, final_tau)

best_tour, best_cost, history, final_tau = st.session_state['aco_result']
tour_names = [CITIES[c] for c in best_tour] + [CITIES[best_tour[0]]]
tour_str   = "  →  ".join(tour_names)

# ── Metrics row ──────────────────────────────
st.markdown('<div class="section-header">🏆 Solution</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Best Tour Cost</div>
        <div class="value">{int(best_cost)}</div>
        <div class="sub">total distance</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Iterations Run</div>
        <div class="value">{n_iter}</div>
        <div class="sub">generations</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Ants Used</div>
        <div class="value">{n_ants}</div>
        <div class="sub">per iteration</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Rank Window (w)</div>
        <div class="value">{w}</div>
        <div class="sub">elite ants</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f'<div class="route-box">🔴 OPTIMAL ROUTE &nbsp;&nbsp; {tour_str}</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Charts row ───────────────────────────────
ch1, ch2 = st.columns(2)

with ch1:
    st.markdown('<div class="section-header">📈 Convergence Curve</div>', unsafe_allow_html=True)
    iters      = [h['iteration']  for h in history]
    best_costs = [h['best_cost']  for h in history]
    iter_bests = [h['iter_best']  for h in history]

    fig_conv, ax = plt.subplots(figsize=(5.5, 3.2))
    fig_conv.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    ax.plot(iters, iter_bests, color='#3949ab', lw=1.2, alpha=0.5, label='Iteration Best')
    ax.plot(iters, best_costs, color='#00bcd4', lw=2.2, label='Global Best')
    ax.fill_between(iters, best_costs, min(best_costs)*0.98, alpha=0.1, color='#00bcd4')
    ax.set_xlabel("Iteration", color='#546e7a', fontsize=10)
    ax.set_ylabel("Tour Cost",  color='#546e7a', fontsize=10)
    ax.tick_params(colors='#546e7a')
    for spine in ax.spines.values(): spine.set_edgecolor('#1a2540')
    ax.legend(framealpha=0, labelcolor='#90a4ae', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_conv)
    plt.close()

with ch2:
    st.markdown('<div class="section-header">🗺️ Best Route Visualization</div>', unsafe_allow_html=True)
    # Place cities in a circle
    angles = [2 * np.pi * i / N for i in range(N)]
    cx = [np.cos(a) for a in angles]
    cy = [np.sin(a) for a in angles]

    fig_route, ax = plt.subplots(figsize=(4.5, 3.5))
    fig_route.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')

    # Draw edges of best tour
    for i in range(N):
        a = best_tour[i]; b = best_tour[(i+1) % N]
        ax.annotate("", xy=(cx[b], cy[b]), xytext=(cx[a], cy[a]),
                    arrowprops=dict(arrowstyle="-|>", color='#00bcd4', lw=1.8))

    # Draw nodes
    for i in range(N):
        ax.scatter(cx[i], cy[i], s=280, color='#3949ab', zorder=5, edgecolors='#7986cb', lw=1.5)
        ax.text(cx[i]*1.22, cy[i]*1.22, CITIES[i], ha='center', va='center',
                color='#a5d8ff', fontsize=12, fontweight='bold',
                fontfamily='monospace')

    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig_route)
    plt.close()

# ── Pheromone heatmap + iteration log ────────
ph1, ph2 = st.columns(2)

with ph1:
    st.markdown('<div class="section-header">🧪 Final Pheromone Matrix</div>', unsafe_allow_html=True)
    fig_tau, ax = plt.subplots(figsize=(4.5, 3.2))
    fig_tau.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    im2 = ax.imshow(final_tau, cmap='plasma', aspect='auto')
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(CITIES, color='#a5d8ff', fontsize=11)
    ax.set_yticklabels(CITIES, color='#a5d8ff', fontsize=11)
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{final_tau[i][j]:.1f}", ha='center', va='center',
                    color='white', fontsize=8)
    ax.tick_params(colors='#546e7a')
    for spine in ax.spines.values(): spine.set_edgecolor('#1a2540')
    fig_tau.colorbar(im2, ax=ax).ax.yaxis.set_tick_params(color='#546e7a')
    plt.tight_layout()
    st.pyplot(fig_tau)
    plt.close()

with ph2:
    st.markdown('<div class="section-header">📋 Iteration Log (last 15)</div>', unsafe_allow_html=True)
    log_data = history[-15:]
    global_min = min(h['best_cost'] for h in history)
    for h in log_data:
        is_best = h['best_cost'] == global_min
        cls = "iter-row best-row" if is_best else "iter-row"
        tour_s = "→".join([CITIES[c] for c in h['best_tour']])
        st.markdown(f"""
        <div class="{cls}">
            <span>Iter {h['iteration']:03d}</span>
            <span>{tour_s}</span>
            <span style='color:{"#00bcd4" if is_best else "#7986cb"}'>{int(h['best_cost'])}</span>
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center; font-family:Space Mono,monospace; font-size:11px; color:#263238;'>
RANK-BASED ANT SYSTEM · RAS · TSP · 5 CITIES
</p>
""", unsafe_allow_html=True)
