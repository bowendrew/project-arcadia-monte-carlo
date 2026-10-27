import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Garamond", "EB Garamond", "Georgia", "Times New Roman"],
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

N, SEED, BUDGET = 10_000, 42, 130.0   # simulations, random seed, case budget ($M)
rng = np.random.default_rng(SEED)

# ---- Base-case model (same numbers as the slides) ----
REGIONAL_POOL = 11_000_000                                   # Exhibit 1 regional visitors
capture = np.array([0.030, 0.040, 0.050, 0.055, 0.055])      # regional capture path, Years 1-5
dest = np.array([300, 375, 375, 375, 375]) * 1_000           # destination/local visits
spend = np.array([55, 58, 61, 63, 65])                       # $ per visit
care = 5.6                                                   # $M/yr, Exhibit 2 (Fairy, Unicorn, Mermaid, Centaur)
other_fix = np.array([11.0, 12.0, 14.0, 14.0, 14.0])         # $M/yr: staff, upkeep, marketing, reinvestment
LAND, CREATURES, LEAD = 24.0, 26.0, 7.0                      # $M: Exhibits 1-2 and staff-lead placeholder


def run(f, g, s, constr, m, v):
    """f = capture multiplier, g = destination multiplier, s = spend multiplier,
    constr = construction cost ($M), m = other-fixed-cost multiplier, v = variable cost rate.
    Returns cumulative cash for Years 0-5, revenue by year, and visits by year."""
    visits = capture[None, :] * f[:, None] * REGIONAL_POOL + dest[None, :] * g[:, None]
    rev = visits * spend[None, :] * s[:, None] / 1e6
    net = rev * (1 - v[:, None]) - care - other_fix[None, :] * m[:, None]
    y0 = -(LAND + CREATURES + LEAD + constr)
    cum = y0[:, None] + np.cumsum(net, axis=1)
    return np.column_stack([y0, cum]), rev, visits


# Sanity check: all multipliers = 1 should reproduce the slides (Year 5 cumulative cash about +39)
one = np.ones(1)
chk, _, _ = run(one, one, one, np.array([35.0]), one, np.array([0.15]))
print("Deterministic base case, cumulative cash Years 0-5 ($M):", np.round(chk[0], 1))

# ---- Uncertain inputs: triangular(low, most likely, high) ----
tri = lambda lo, mode, hi: rng.triangular(lo, mode, hi, N)
f = tri(0.40, 1.00, 1.45)        # capture-rate multiplier (about 2.2%-8.0% by Year 5)
g = tri(0.30, 1.00, 1.15)        # destination-visit multiplier
s = tri(0.75, 1.00, 1.10)        # spend-per-visit multiplier
constr = tri(30.0, 35.0, 50.0)   # construction cost $M (skewed toward overruns)
m = tri(0.90, 1.00, 1.25)        # other fixed operating costs multiplier
v = tri(0.12, 0.15, 0.20)        # variable cost share of revenue

cum, rev, visits = run(f, g, s, constr, m, v)
y5 = cum[:, 5]
peak_need = -cum.min(axis=1)     # deepest cumulative cash hole ($M)
be_year = np.where((cum[:, 1:] >= 0).any(axis=1), (cum[:, 1:] >= 0).argmax(axis=1) + 1, 99)

pct = lambda a, q: np.percentile(a, q)
res = {
    "P(cumulative cash >= 0 by Year 5)": (y5 >= 0).mean(),
    "P(break even by Year 4)": (be_year <= 4).mean(),
    "P(break even by Year 3)": (be_year <= 3).mean(),
    "P(peak funding need > $130M budget)": (peak_need > BUDGET).mean(),
    "Year 5 cumulative cash P10 ($M)": pct(y5, 10),
    "Year 5 cumulative cash P50 ($M)": pct(y5, 50),
    "Year 5 cumulative cash P90 ($M)": pct(y5, 90),
    "Year 5 attendance P10 (K)": pct(visits[:, 4], 10) / 1e3,
    "Year 5 attendance P50 (K)": pct(visits[:, 4], 50) / 1e3,
    "Year 5 attendance P90 (K)": pct(visits[:, 4], 90) / 1e3,
    "5-yr revenue P50 ($M)": pct(rev.sum(axis=1), 50),
}
for k, val in res.items():
    print(f"{k}: {val:.1%}" if k.startswith("P(") else f"{k}: {val:,.1f}")

# ---- Extra statistics: operating cash flow and payback beyond Year 5 ----
net_cash = np.diff(cum, axis=1)   # annual operating cash, Years 1-5
n5 = net_cash[:, 4]
print(f"P(Year 5 operating cash > 0): {(n5 > 0).mean():.1%}")
print(f"P(operating cash > 0 in every year 2-5): {(net_cash[:, 1:] > 0).all(axis=1).mean():.1%}")
print("Year 5 annual operating cash P10 / P50 / P90 ($M): "
      f"{pct(n5, 10):.1f} / {pct(n5, 50):.1f} / {pct(n5, 90):.1f}")
for yr in (6, 7, 8):   # assumes Year 5 operating cash simply continues (an extrapolation)
    print(f"P(build recovered by Year {yr}, flat Year 5 cash): {(cum[:, 5] + (yr - 5) * n5 >= 0).mean():.1%}")
print(f"Peak funding need P50 / P90 / max ($M): {pct(peak_need, 50):.1f} / {pct(peak_need, 90):.1f} / {peak_need.max():.1f}")

# ---- Charts ----
PINK_LIGHT, PINK, PINK_DARK, ROSE = "#f2a7c8", "#e0709f", "#a8325f", "#7a1f46"

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].hist(y5, bins=60, color=PINK, alpha=0.9)
ax[0].axvline(0, color=ROSE, lw=2)
ax[0].axvline(np.median(y5), color="k", ls="--", lw=1.5)
ax[0].set_xlabel("Year 5 Cumulative Cash ($M)")
ax[0].set_ylabel("Simulations")
ax[0].set_title("Year 5 Outcomes")

yrs = np.arange(6)
for lo, hi, a in [(10, 90, 0.35), (25, 75, 0.6)]:
    ax[1].fill_between(yrs, np.percentile(cum, lo, axis=0), np.percentile(cum, hi, axis=0),
                       color=PINK, alpha=a, lw=0)
ax[1].plot(yrs, np.percentile(cum, 50, axis=0), color=PINK_DARK, lw=3, label="Median")
ax[1].axhline(0, color=ROSE, lw=2)
ax[1].axhline(-BUDGET, color="gray", ls=":", lw=2, label="$130M budget")
ax[1].set_xlabel("Year")
ax[1].set_ylabel("Cumulative Cash ($M)")
ax[1].set_title("Cumulative Cash, 10th-90th PCT")
ax[1].legend()

plt.tight_layout()
plt.savefig("arcadia_monte_carlo.png", dpi=300)
print("Saved arcadia_monte_carlo.png")

years = np.arange(6)
cum_costs = np.array([92.0, 113.8, 138.5, 166.6, 195.4, 224.6])
cum_rev = np.array([0, 34.65, 81.92, 138.35, 200.09, 263.79])

plt.figure(figsize=(8, 4))
plt.plot(years, cum_costs, color="#dd7dc8", lw=4, marker="o", ms=10, label="Total Costs")
plt.plot(years, cum_rev, color="#6a1b9a", lw=4, marker="o", ms=10, label="Total Revenue")
plt.ylim(0, 300)
plt.yticks(range(0, 301, 50), [f"${v}M" for v in range(0, 301, 50)])
plt.xlabel("Year")
plt.ylabel("Running Total ($M)")
plt.grid(axis="y", color="#dddddd")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("arcadia_breakeven.png", dpi=150)