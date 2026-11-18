<div align="center">

# VaR-Based Portfolio Construction with GARCH Models

**Estimating market volatility with GARCH models to optimise portfolio returns while minimising Value at Risk**

`NIFTY 50` · `S&P 500` · `Euro Stoxx 50`

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![arch](https://img.shields.io/badge/models-arch-orange)
![Data](https://img.shields.io/badge/data-Yahoo%20Finance-6001D2)

</div>

---

## Overview

**Volatility clustering** is a phenomenon observed in time series such as the closing prices of stock market indices: periods of high volatility tend to be followed by more high volatility, and calm periods by more calm.

This makes market risk hard to assess. **Autoregressive Conditional Heteroskedasticity (ARCH)** models address this by modelling the variance itself. Unlike standard regression models, which assume constant variance (homoskedasticity), they let variance change over time, which is what allows them to capture volatility clustering.

To design optimal risk-averse portfolios, the chosen risk metric is **Value at Risk (VaR)**: the share of the invested sum that could be lost on a given market day at a chosen confidence level. VaR is estimated from volatility forecasts produced by **generalised ARCH (GARCH)** models.

### Goal

Build an optimal portfolio that splits capital across three markets so that **VaR is minimised** while a **fixed year-on-year return** is achieved:

| Market | Index | Yahoo ticker |
|:--|:--|:--|
| India | NIFTY 50 | `^NSEI` |
| United States | S&P 500 | `^GSPC` |
| Eurozone | Euro Stoxx 50 | `^STOXX50E` |

### Roadmap

- [x] **Stage 1:** GARCH model fit and volatility estimation for each index: NIFTY 50, S&P 500 and Euro Stoxx 50
- [ ] **Stage 2:** Model how correlation between the three markets changes over time (DCC-GARCH)
- [x] **Stage 3:** Minimum-VaR portfolio for a target annual return, built once on the training data (`portoflio.ipynb`)
- [ ] **Stage 4:** Model volatility for each **constituent** of the indices, not only the indices themselves
- [x] **Stage 5:** Dynamic portfolio rebalancing as risk estimates change (`dynamic-portfolio.ipynb`)

| Window | Period |
|:--|:--|
| Training and portfolio construction | earliest available → 2017 |
| Out-of-sample test | 2018 → 2025 |

> [!NOTE]
> The project plan uses training data from 1990, but Yahoo Finance's daily history is shorter for two of the indices: `^NSEI` (NIFTY 50) begins on **17 Sep 2007** and `^STOXX50E` (Euro Stoxx 50) on **30 Mar 2007**. Their training windows therefore run from 2007 to 2017. The S&P 500 uses the full 1990–2017 window.

---

## Stage 1: GARCH Model Fit and Volatility Estimation

| Index | Notebook | Training window | Test window |
|:--|:--|:--|:--|
| NIFTY 50 | [`nifty.ipynb`](nifty.ipynb) | 2007-09-18 → 2017-12-29 (2,514 days) | 2018 → 2025 (1,972 days) |
| S&P 500 | [`sp500.ipynb`](sp500.ipynb) | 1990-01-03 → 2017-12-29 (7,055 days) | 2018 → 2025 (2,011 days) |
| Euro Stoxx 50 | [`eurostoxx50.ipynb`](eurostoxx50.ipynb) | 2007-04-02 → 2017-12-29 (2,692 days) | 2018 → 2025 (2,008 days) |

All three notebooks follow the same steps, so the results can be compared directly.

### Pipeline

```text
Yahoo Finance (^NSEI, ^GSPC, ^STOXX50E)
   └─▶ daily log returns (%)
         └─▶ stylised-fact tests (Jarque–Bera, Ljung–Box, ARCH-LM)
               └─▶ fit 4 models on the training window ─▶ pick lowest BIC
                     └─▶ 1-day-ahead σ forecasts for 2018–2025
                           └─▶ 95% / 99% VaR ─▶ Kupiec + Christoffersen backtests
```

### Models compared

All models use a constant mean, $r_t = \mu + \sigma_t z_t$.

| Model | Variance equation | Innovations |
|:--|:--|:--|
| GARCH(1,1) | $\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2$ | Normal / Student-t |
| GJR-GARCH(1,1,1) | $\sigma_t^2 = \omega + (\alpha + \gamma\mathbb{1}_{\varepsilon_{t-1}<0})\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2$ | Normal / Student-t |

### Key results

**Stylised facts.** All three return series have fat tails (training-window excess kurtosis: NIFTY 11.8, S&P 9.0, Euro Stoxx 5.5) and strong ARCH effects (Ljung–Box on squared returns and ARCH-LM both give p ≈ 0). A constant-variance model is not appropriate for any of them.

**Model selection.** For every index, **GJR-GARCH(1,1,1) with Student-t innovations** has the lowest AIC and BIC. Its standardised residuals show no remaining ARCH effects in any market.

| Parameter (GJR-t) | NIFTY 50 | S&P 500 | Euro Stoxx 50 |
|:--|--:|--:|--:|
| γ (leverage) | 0.117 | 0.146 | 0.198 |
| β | 0.920 | 0.917 | 0.892 |
| Persistence | 0.991 | 0.990 | 0.991 |
| Half-life of a volatility shock | 80 days | 67 days | 75 days |
| ν (t degrees of freedom) | 8.1 | 7.0 | 7.7 |
| Long-run volatility (p.a.) | 20.8% | 15.9% | 27.5% |
| ARCH-LM p on std. residuals | 0.69 | 0.53 | 0.34 |

The leverage effect is strongest in Europe: a negative shock there raises next-day variance far more than a positive shock of the same size.

**Out-of-sample VaR backtest, 2018–2025.** Number of days on which the loss exceeded VaR, with the Kupiec p-value in brackets.

| Model | NIFTY 50 95% | NIFTY 50 99% | S&P 500 95% | S&P 500 99% | Euro Stoxx 95% | Euro Stoxx 99% |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| *Expected* | *98.6* | *19.7* | *100.6* | *20.1* | *100.4* | *20.1* |
| **GJR-t** | **97** (0.87) | **25** (0.25) | **117** (0.10) | **33** (0.01†) | **101** (0.95) | **28** (0.09) |
| GJR-N | 93 (0.56) | 30 (0.03†) | 111 (0.29) | 50 (0.00†) | 89 (0.23) | 37 (0.00†) |
| GARCH-t | 108 (0.34) | 28 (0.08) | 125 (0.02†) | 41 (0.00†) | 120 (0.05) | 34 (0.01†) |
| GARCH-N | 103 (0.65) | 33 (0.01†) | 121 (0.04†) | 53 (0.00†) | 110 (0.33) | 42 (0.00†) |
| Historical sim. (250-day) | 107 (0.39) | 31 (0.02†) | 116 (0.12) | 37 (0.00†) | 123 (0.03†) | 35 (0.00†) |

<sub>† means correct coverage is rejected at the 5% level. Full tables, including the Christoffersen independence test, are in each notebook.</sub>

> **Takeaways**
> - GJR-t's breach count is the closest to the expected number in five of the six cases (the exception is the S&P 500 at 95%, where GJR-N is closer), and it passes the coverage test in five of six cases.
> - The one failure is the S&P 500 at 99% (33 breaches against 20 expected). The US training window (1990–2017) was calmer than the 2018–2025 test period, which included the 2020 crash, the 2022 rate-hike sell-off and the April 2025 tariff shock, so the model's tails are too thin for it. Re-estimating the parameters periodically, as planned for the dynamic stage, should help.
> - Historical simulation is the worst choice for portfolio work: its breaches are strongly clustered in all three markets (Christoffersen p < 0.01), because it reacts late to a crash and then overstates risk for a full year.

Each notebook writes its daily σ and VaR forecasts to `data/<index>_garch_forecasts.csv` for use in the portfolio stages.

### Value at Risk by index

The figures below are the one-day VaR forecasts from each index's GJR-t model over the 2018–2025 test window. They are stated as the percentage of the position that could be lost in one day. A 99% VaR of 2.5% means that on 99 days out of 100, the index should not fall by more than 2.5%.

| One-day VaR, 2018–2025 | NIFTY 50 | S&P 500 | Euro Stoxx 50 |
|:--|--:|--:|--:|
| Average 95% VaR | 1.57% | 1.67% | 1.83% |
| Average 99% VaR | 2.47% | 2.66% | 2.87% |
| Median 99% VaR | 2.15% | 2.20% | 2.47% |
| Typical range of 99% VaR (10th–90th percentile) | 1.53 – 3.42% | 1.37 – 4.28% | 1.70 – 4.20% |
| Peak 99% VaR (COVID crash) | 16.6% (24 Mar 2020) | 17.7% (17 Mar 2020) | 18.1% (13 Mar 2020) |
| 99% VaR at the end of 2025 | 1.46% | 1.70% | 1.71% |
| Average extra loss on days VaR was breached | 0.84 pp | 0.87 pp | 0.91 pp |

**Average 99% VaR by year**

| Year | NIFTY 50 | S&P 500 | Euro Stoxx 50 |
|:--|--:|--:|--:|
| 2018 | 2.13% | 2.51% | 2.52% |
| 2019 | 2.15% | 2.11% | 2.29% |
| 2020 | 4.20% | 4.38% | 4.56% |
| 2021 | 2.41% | 2.01% | 2.51% |
| 2022 | 2.76% | 3.74% | 3.61% |
| 2023 | 1.77% | 2.06% | 2.50% |
| 2024 | 2.31% | 1.94% | 2.40% |
| 2025 | 1.97% | 2.56% | 2.57% |

#### NIFTY 50

The NIFTY 50 had the **lowest average VaR** of the three: 2.47% at 99%, or a 1-in-100-day loss of about 2.5%. This is lower than its long-run level of 20.8% annualised volatility, because 2018–2025 was calmer for Indian equities than the 2007–2017 training period, which included the 2008 crisis. Risk stayed close to its baseline outside two episodes:

- **COVID crash.** 99% VaR rose from 2.3% in mid-February 2020 to a peak of 16.6% on 24 March. The worst day, a 13.9% fall on 23 March, broke even the elevated 11.3% VaR. Ten of the 25 breaches at 99% fell in 2020.
- **4 June 2024 election-result day.** The index fell 6.1% when the 99% VaR was only 2.1%, the largest breach outside 2020.

By 2023 the NIFTY's 99% VaR had fallen to 1.8%, the lowest annual average of any index in any year. It ended 2025 at 1.46%. Breaches were spread over time (Christoffersen p = 0.42), so the model's risk estimates can be relied on for allocation.

#### S&P 500

The S&P 500's average VaR was in the middle (2.66% at 99%), but it had the **widest swings**: its 99% VaR ranged from under 1% in January 2018 to 17.7% in March 2020, and the gap between its 10th and 90th percentiles is the largest of the three. US risk is the most regime-dependent: very low in calm markets, very high in stress.

- **2022 was the S&P's second-worst year**, with an average 99% VaR of 3.74% as rate hikes hit growth stocks, compared with 2.76% for the NIFTY.
- **2025 VaR rose again** to 2.56%, driven by the April tariff sell-off.
- **This is the one VaR that fails its backtest.** There were 33 breaches at 99% against about 20 expected (1.64% of days). They were spread across every year rather than concentrated in 2020, so the model's tails are too thin in general and not only in the crash. In practice the S&P 500's true 99% VaR is somewhat higher than the model reports. The 95% VaR is acceptable (5.8% breach rate, Kupiec p = 0.10).

#### Euro Stoxx 50

The Euro Stoxx 50 is the **riskiest of the three markets** by VaR. It has the highest average (2.87% at 99%), the highest COVID peak (18.1% on 13 March 2020, the earliest of the three peaks) and the highest floor: its 99% VaR never fell below 1.37%. It also has the strongest leverage effect (γ = 0.198), so its VaR rises fastest after a fall.

- **2022** added a second stress period (average 99% VaR of 3.61%), driven by the war in Ukraine and the energy-price shock.
- **The VaR is well calibrated.** At 95% there were 101 breaches against 100.4 expected (Kupiec p = 0.95, Christoffersen p = 0.97). At 99% there were 28 against 20.1 expected (p = 0.09).
- **Breaches were the largest.** On days the loss exceeded the 99% VaR, it did so by an average of 0.91 percentage points, slightly more than in the other two markets.

#### Implications for the portfolio

- **Ranking by risk.** On average the Euro Stoxx 50 carries the most one-day risk, then the S&P 500, then the NIFTY 50. A minimum-VaR portfolio will tend to favour the NIFTY, subject to the return target.
- **Risk changes over time.** Each market's VaR moved by a factor of more than 10 between calm and crisis periods, and the ranking changed from year to year: the S&P was the safest market in 2021 and 2024 and among the riskiest in 2022 and 2025. A portfolio whose weights are fixed once will therefore be poorly hedged for much of the test period, which supports the planned dynamic rebalancing stage.
- **Stress arrives together.** All three VaRs peaked within 11 days of each other in March 2020, so diversification across the three markets gives the least protection exactly when it is needed. This is why Stage 2 models the time-varying correlation between the markets.
- **The US estimate needs a cushion.** Until the model is re-estimated on more recent data, the S&P 500's 99% VaR should be treated as understated when it feeds into the optimiser.

---

## Year-on-Year Returns vs Value at Risk

**Notebook:** [`yoy-returns.ipynb`](yoy-returns.ipynb)

This analysis asks whether risk, as measured by VaR, is related to returns. Each index's calendar-year price return is compared with two VaR measures from its GJR-t model:

- **Average VaR during the year** (contemporaneous): are high-risk years also good or bad years?
- **VaR on the first trading day of the year** (predictive): does the risk level at the start of a year say anything about the return over that year?

The Stage 1 notebooks only export VaR for 2018–2025, which gives 8 years per index. To get more years, this notebook re-estimates the same GJR-t model on the same training window and runs it over the full history with fixed parameters. This gives **2008–2025 (18 years)** for the NIFTY 50 and Euro Stoxx 50 and **1991–2025 (35 years)** for the S&P 500. Years before 2018 are in-sample and are marked separately. Over 2018–2025 the recomputed VaR matches the Stage 1 exports exactly.

### Correlation results (99% VaR)

| VaR measure | Sample | NIFTY 50 | S&P 500 | Euro Stoxx 50 | Pooled |
|:--|:--|:-:|:-:|:-:|:-:|
| Same-year average VaR | All years | −0.36 | −0.56† | **−0.67†** | **−0.52†** |
| | Test 2018–25 | +0.02 | −0.53 | −0.56 | −0.45† |
| Start-of-year VaR | All years | +0.34 | +0.10 | +0.33 | +0.18 |
| | Test 2018–25 | +0.35 | +0.35 | +0.54 | +0.35 |

<sub>Pearson r. † means significant at the 5% level. The notebook also reports Spearman rank correlations, which agree in sign throughout. With 8 years, |r| has to exceed about 0.7 to be significant.</sub>

### Findings

**High-risk years are bad years.** Average VaR and the same year's return are negatively correlated in every market over the full sample. Pooled across the three indices the correlation is −0.52, significant both over all years and within the test window.

- **Euro Stoxx 50:** the strongest and most consistent link (r = −0.67, Spearman ρ = −0.56, both significant). Three of its four highest-VaR years (2008, 2011 and 2020) were losing years. The exception was the 2009 recovery.
- **S&P 500:** the linear correlation is significant (r = −0.56), but the rank correlation is not (ρ = −0.28). The relationship is driven by a few crash years (2002, 2008, 2022) rather than holding steadily from year to year.
- **NIFTY 50:** only a weak link, and none over 2018–2025 (r = +0.02). Its highest-VaR test year, 2020, still ended up 14.9% because of the sharp post-crash recovery.

This is the leverage effect seen at a yearly scale: falling prices push volatility up, so high risk and poor returns arrive together. Because both are measured over the same year, it cannot be used to predict returns.

**High VaR at the start of a year is followed by better returns, but the evidence is weak.** Every index shows a positive correlation between start-of-year VaR and the following year's return (+0.10 to +0.54), but none is significant. The closest is the pooled test-window result (Spearman ρ = +0.39, p = 0.06). The pattern comes mostly from years that began just after a sell-off. In 2009, VaR was high on 1 January and the NIFTY then rose 76%, the S&P 23% and the Euro Stoxx 21%. In 2019, after the late-2018 sell-off, the S&P rose 29% and the Euro Stoxx 26%. This fits a risk premium or mean reversion after crashes, but with 8–35 years per index a single year such as 2009 moves the result a lot.

**The daily rolling view agrees.** Over 2018–2025, daily 99% VaR has a correlation of −0.32 to −0.46 with the *trailing* one-year return and +0.43 to +0.51 with the *forward* one-year return. These windows overlap almost completely from one day to the next, so the figures are illustrative rather than independent evidence.

**Higher VaR was not rewarded across markets.**

| 2018–2025 | NIFTY 50 | S&P 500 | Euro Stoxx 50 |
|:--|--:|--:|--:|
| Mean annual return | 12.2% | 13.8% | 7.6% |
| Mean 99% VaR | 2.46% | 2.66% | 2.87% |
| Return per unit of VaR | 5.3 | 6.7 | 3.4 |

The riskiest market, the Euro Stoxx 50, earned the lowest return, and the S&P 500 earned the most return per unit of risk.

### What this means for the portfolio

- **Reducing VaR does not have to cost return.** Across these three markets, the one with the highest VaR also had the lowest return. A minimum-VaR portfolio that underweights the Euro Stoxx 50 should give up little return to meet its target.
- **Within a market, cutting exposure when VaR is high has a cost.** If high starting VaR is followed by stronger returns, a dynamic strategy that reduces exposure after a crash may miss the recovery, as in 2009 and 2020. The rebalancing rule in Stage 5 should be tested for this.
- **Caveats.** Returns are price returns and exclude dividends, which are worth roughly 1–3% a year. VaR before 2018 comes from a model fitted on those same years. The yearly samples are small, so the predictive results are suggestive rather than established.

---

## Stage 3: Minimum-VaR Portfolio

**Notebook:** [`portoflio.ipynb`](portoflio.ipynb)

**$10,000** is split across the three indices once, at the start of 2018, and held to the end of 2025. The weights minimise the portfolio's one-day 99% VaR subject to a **target year-on-year expected return**, which is a parameter at the top of the notebook.

```python
INITIAL_INVESTMENT = 10_000    # dollars
TARGET_RETURN      = 0.08      # target expected year-on-year return
VAR_LEVEL          = 0.99      # VaR confidence level used in the optimisation
ALLOW_SHORT        = False     # weights between 0 and 1
```

### Method

Only training-period information (to 29 Dec 2017) is used to choose the weights:

| Input | How it is estimated |
|:--|:--|
| Expected return | Training-period mean daily return × 252: NIFTY 10.6%, S&P 7.7%, Euro Stoxx 1.2% a year |
| Volatility for 2018–2025 | Multi-step GJR-GARCH-t forecast made at the end of 2017, averaged over the test horizon: 20.3%, 15.6% and 26.9% a year |
| Correlation | Training-period correlation of the GARCH standardised residuals (constant conditional correlation) |
| Tail quantile | Student-t, with ν equal to the weight-averaged ν of the three models |

$$\min_w \; -\left(w^\top\mu - q_\nu\sqrt{w^\top D R D\, w}\right) \quad \text{s.t.} \quad w^\top\mu^{annual} \ge \text{target}, \;\; \textstyle\sum w_i = 1, \;\; w_i \ge 0$$

The test period is used only for evaluation. Each day, the GARCH models (with parameters fixed at their training values) give a one-step-ahead volatility for each index. The portfolio VaR is then computed from the current, drifted weights and reported in percent and in dollars. The portfolio is compared with putting the whole $10,000 into each single index.

The highest target that can be reached without short selling is 10.6%, the NIFTY 50's training-period return.

### Results (default 8% target)

**Allocation:** 33.5% NIFTY 50 ($3,348), 66.5% S&P 500 ($6,652), 0% Euro Stoxx 50.

The 8% target is not binding. The lowest-VaR mix already has an expected return of 8.65%, so any target up to 8.65% gives the same weights. The Euro Stoxx 50 is left out because, on the training data, it had the highest forecast volatility, the lowest expected return and a 0.63 correlation with the S&P 500.

| $10,000 invested 2018–2025 | Min-VaR portfolio | 100% NIFTY 50 | 100% S&P 500 | 100% Euro Stoxx 50 |
|:--|--:|--:|--:|--:|
| **Return** | | | | |
| Final value | $25,339 | $24,813 | **$25,604** | $16,542 |
| Annualised return | 12.0% | 11.7% | **12.1%** | 6.3% |
| Worst calendar year | −11.9% | **+3.2%** | −19.4% | −14.8% |
| **Risk** | | | | |
| Average forecast 99% VaR | **2.08%** | 2.34% | 2.58% | 2.78% |
| Average forecast 99% VaR ($) | **$310** | $352 | $383 | $311 |
| Peak forecast 99% VaR ($) | **$1,191** | $1,227 | $1,606 | $1,311 |
| Actual worst 1% of days | **2.56%** | 2.67% | 3.41% | 3.64% |
| Annualised volatility | **15.2%** | 16.6% | 19.3% | 18.6% |
| Maximum drawdown | −35.0% | −38.4% | **−33.9%** | −38.3% |
| VaR breaches (expected ≈ 21) | 39† | 25 | 33† | 28 |
| **Risk-adjusted** | | | | |
| Sharpe ratio (rf = 0) | **0.82** | 0.75 | 0.69 | 0.42 |
| Return ÷ average VaR | **5.75** | 4.98 | 4.69 | 2.27 |

<sub>Bold marks the best value in each row. † means the Kupiec test rejects the VaR at the 5% level.</sub>

### Findings

- **Same return, less risk.** The portfolio returned almost as much as the best single index (12.0% a year against 12.1% for the S&P 500). Its average VaR was 19% lower than the S&P 500's and 11% lower than the NIFTY 50's. It also had the lowest volatility and the highest Sharpe ratio.
- **It beat its target.** It returned 12.0% a year against an 8% target, mainly because the S&P 500 did much better over 2018–2025 than its training-period average of 7.7%.
- **Not the best on every measure.** Holding only the S&P 500 had a slightly smaller maximum drawdown (33.9% against 35.0%), because the NIFTY part of the portfolio fell further in March 2020. Holding only the NIFTY 50 never had a losing calendar year.
- **Leaving out the Euro Stoxx 50 was right.** It had the highest forecast risk and returned only 6.3% a year. The portfolio built on training data excluded it without any knowledge of the test period.
- **Raising the target adds risk without adding return.** Targets above 8.65% shift weight to the NIFTY 50 (79% NIFTY at a 10% target). This raises planned VaR from 2.17% to 2.68%, while the return over 2018–2025 slipped slightly (11.8% at a 10% target).

### Calibration caveat

The portfolio VaR was breached on 39 days (1.88%) against about 21 expected. The volatility forecast is accurate: the standardised portfolio returns have a standard deviation of 0.97. The tail is the problem. The worst 1% of standardised returns reached −2.87, against −2.52 assumed by the weight-averaged t distribution, so the 99% VaR is understated by about 14%.

There are two causes. First, two-thirds of the portfolio is in the S&P 500, whose own 99% VaR already fails its backtest. Second, markets fall together more in a crash than their average correlation suggests. Scaled up by 14%, the portfolio's average VaR would be about 2.37%. That is level with the NIFTY 50 alone and still below the other two indices. On actual worst-day losses, which do not depend on any model, the portfolio remains the safest of the four.

Stage 2 (DCC-GARCH, time-varying correlations) and a fatter-tailed or filtered-historical-simulation quantile would address this.

**Assumptions.** Returns are in local currency (as if currency-hedged) and exclude dividends. There are no transaction costs and no rebalancing.

---

## Stage 5: Dynamic Minimum-VaR Portfolio

**Notebook:** [`dynamic-portfolio.ipynb`](dynamic-portfolio.ipynb)

The same $10,000 and the same 8% target as Stage 3, but the portfolio is **re-optimised every month**. On the last trading day of each month, the **whole accumulated value** is re-split across the three indices so that the forecast VaR for the coming month is as low as possible while still meeting the target.

```python
REBALANCE_FREQ       = "M"            # "W", "M" or "Q"
REFIT_GARCH          = True           # re-estimate GARCH parameters at every rebalance
CORR_WINDOW          = 500            # trading days of residuals for correlations (None = expanding)
RETURN_WINDOW        = None           # expanding-window expected returns
IF_INFEASIBLE        = "max_return"   # what to do if no mix can reach the target
TRANSACTION_COST_BPS = 0.0
```

### What changes at each rebalance

Only data up to that day's close is used:

| Input | Static (Stage 3) | Dynamic (Stage 5) |
|:--|:--|:--|
| GARCH parameters | fitted once, to 2017 | re-estimated each month on all data to date |
| Volatility forecast | average over 2018–2025, made in Dec 2017 | average over the next month, made that day |
| Correlations | training-period residuals | last 500 trading days of residuals |
| Expected returns | training-period mean | mean of all returns to date |
| Weights | set once, then drift | reset to the new optimum each month |

If no long-only mix can reach the target, the portfolio moves to the index with the highest expected return (`max_return`) or to the lowest-VaR mix regardless of return (`min_var`). This happened once at the 8% target, at the end of March 2020.

The same engine with no rebalancing reproduces Stage 3 exactly ($25,339). This confirms that the two notebooks are directly comparable.

### Results (8% target, monthly rebalancing)

**Average allocation:** 48% NIFTY 50, 43% S&P 500, 8% Euro Stoxx 50. Average one-way turnover was about 20% per month.

| $10,000 invested 2018–2025 | Dynamic | Static | 100% NIFTY 50 | 100% S&P 500 | 100% Euro Stoxx 50 |
|:--|--:|--:|--:|--:|--:|
| **Return** | | | | | |
| Final value | $24,861 | $25,339 | $24,813 | **$25,604** | $16,542 |
| Annualised return | 11.7% | 12.0% | 11.7% | **12.1%** | 6.3% |
| Worst calendar year | −5.5% | −11.9% | **+3.2%** | −19.4% | −14.8% |
| **Risk** | | | | | |
| Average forecast 99% VaR | **1.89%** | 2.08% | 2.34% | 2.58% | 2.78% |
| Average forecast 99% VaR ($) | **$271** | $310 | $352 | $383 | $311 |
| Actual worst 1% of days | **2.47%** | 2.56% | 2.67% | 3.41% | 3.64% |
| Annualised volatility | **14.3%** | 15.2% | 16.6% | 19.3% | 18.6% |
| Worst single day | **$861** | $1,049 | $1,310 | $1,215 | $1,028 |
| Maximum drawdown | −36.7% | −35.0% | −38.4% | **−33.9%** | −38.3% |
| VaR breaches (expected ≈ 21) | 37† | 39† | 25 | 33† | 28 |
| **Risk-adjusted** | | | | | |
| Sharpe ratio (rf = 0) | **0.85** | 0.82 | 0.75 | 0.69 | 0.42 |
| Return ÷ average VaR | **6.19** | 5.75 | 4.98 | 4.69 | 2.27 |

<sub>Bold marks the best value in each row. † means the Kupiec test rejects the VaR at the 5% level.</sub>

### Findings

- **Lower risk.** Compared with the static portfolio, dynamic reallocation cut average VaR by 9%, volatility by 6% and the worst single day by 18%. It was the lowest-risk strategy on every measure except maximum drawdown.
- **The biggest gain came in 2022.** The dynamic portfolio lost 5.5% that year, against 11.9% for the static portfolio and 19.4% for the S&P 500, as it moved weight away from US equities while their volatility rose.
- **A small cost in return.** It returned 11.7% a year against 12.0% for the static portfolio, still far above the 8% target. The cost is about 0.26 percentage points a year, mainly from holding less of the S&P 500, which turned out to be the best performer.
- **The best risk-adjusted result of the five strategies,** with a Sharpe ratio of 0.85 and return per unit of VaR of 6.2.
- **Drawdown was not improved.** Monthly rebalancing cannot react within the few weeks of a crash like March 2020.
- **The infeasible-target rule matters in a crash.** In April 2020 the `max_return` rule held 100% NIFTY 50 (planned VaR 13.5%). `min_var` would have spread the money across all three (planned VaR 9.1%) but missed part of the rebound, ending at $23,812.

### Sensitivity

| 8% target | Final value | Annualised return | Average VaR | Sharpe | Target unreachable |
|:--|--:|--:|--:|--:|--:|
| Static | $25,339 | 11.97% | 2.08% | 0.82 | – |
| Weekly rebalancing | $23,483 | 10.94% | 1.85% | 0.78 | 7 weeks |
| **Monthly rebalancing** | $24,861 | 11.71% | 1.89% | **0.85** | 1 month |
| Quarterly rebalancing | $24,836 | 11.69% | 1.98% | 0.82 | 1 quarter |

- **Monthly is the best frequency.** Weekly rebalancing reacts to noise.
- **A 10% target** is unreachable in 16 of 96 months, because only the NIFTY's expected return exceeds 10%, and the risk advantage largely disappears.
- **Transaction costs** of 10 bps per trade reduce the final value to $23,944 (11.2% a year). The Sharpe ratio falls to 0.82, the same as the static portfolio. After realistic costs, the benefit of rebalancing is lower risk rather than better risk-adjusted return.

### Calibration caveat

The volatility forecast is accurate (standardised returns have a standard deviation of 1.00), but the tail is understated by about 15%. This gives 37 VaR breaches against about 21 expected, slightly clustered (Christoffersen p = 0.03). The causes and remedies are the same as in Stage 3: a fatter-tailed or filtered-historical-simulation portfolio quantile, and DCC correlations that rise in a crisis.

---

## Getting started

```bash
pip install numpy pandas scipy matplotlib statsmodels arch yfinance jupyter
```

```bash
jupyter notebook
```

Run the three index notebooks first, then `yoy-returns.ipynb`, `portoflio.ipynb` and `dynamic-portfolio.ipynb`, which read their cached price files. The dynamic notebook takes about 2–3 minutes, because it refits the models at every rebalance. On its first run each notebook downloads prices from Yahoo Finance and saves them to `data/<index>_daily.csv`. Later runs read that file and work offline. Delete it to fetch fresh data.

### Repository structure

```text
.
├── README.md
├── nifty.ipynb                            # Stage 1: NIFTY 50 GARCH + VaR
├── sp500.ipynb                            # Stage 1: S&P 500 GARCH + VaR
├── eurostoxx50.ipynb                      # Stage 1: Euro Stoxx 50 GARCH + VaR
├── yoy-returns.ipynb                      # year-on-year returns vs VaR, correlation tests
├── portoflio.ipynb                        # Stage 3: $10,000 minimum-VaR portfolio vs single indices
├── dynamic-portfolio.ipynb                # Stage 5: monthly re-optimised minimum-VaR portfolio
├── data/
│   ├── nifty50_daily.csv                  # cached closing prices (^NSEI)
│   ├── sp500_daily.csv                    # cached closing prices (^GSPC)
│   ├── eurostoxx50_daily.csv              # cached closing prices (^STOXX50E)
│   └── *_garch_forecasts.csv              # σ and VaR forecasts, 2018–2025
└── Financial Portfolio Adjustment Through VaR Estimation Using GARCH Models.pptx
```

---

## Data source and references

**Data**

- **Yahoo Finance**, daily historical prices, accessed 18 Nov 2026:
  - *NIFTY 50 (^NSEI)*, 17 Sep 2007 – 31 Dec 2025. https://finance.yahoo.com/quote/%5ENSEI/history/
  - *S&P 500 (^GSPC)*, 2 Jan 1990 – 31 Dec 2025. https://finance.yahoo.com/quote/%5EGSPC/history/
  - *EURO STOXX 50 (^STOXX50E)*, 30 Mar 2007 – 30 Dec 2025. https://finance.yahoo.com/quote/%5ESTOXX50E/history/
- Index providers:
  - NIFTY 50: **NSE Indices Ltd.** (National Stock Exchange of India). https://www.niftyindices.com/
  - S&P 500: **S&P Dow Jones Indices LLC**. https://www.spglobal.com/spdji/
  - EURO STOXX 50: **STOXX Ltd.** https://www.stoxx.com/
- Data was retrieved with **yfinance** (R. Aroussi), https://github.com/ranaroussi/yfinance. yfinance is not affiliated with Yahoo, and the data is meant for research and educational use.

**Software**

- K. Sheppard et al., **arch**: ARCH models in Python. https://github.com/bashtage/arch

**Methodology**

- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779–1801.
- Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73–84.
- Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862.
