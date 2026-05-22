"""
Home Improvement Finance Analyzer

Compares the true cost of liquidating stock investments vs. using a variable-rate
HELOC to fund home improvement projects. Analyzes two strategies:
  1. Staggered liquidation (sell stock on the same schedule as HELOC draws)
  2. HELOC (borrow with variable rates tracking prime, pay from stock sales)
"""

import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HISTORICAL_PRIME_RATES = {
    2011: 0.0325, 2012: 0.0325, 2013: 0.0325, 2014: 0.0325, 2015: 0.0350,
    2016: 0.0375, 2017: 0.0450, 2018: 0.0550, 2019: 0.0475, 2020: 0.0325,
    2021: 0.0325, 2022: 0.0750, 2023: 0.0850, 2024: 0.0750, 2025: 0.0675,
}

DEFAULT_DRAW_SCHEDULE = [(0, 125_000), (1, 62_500), (2, 62_500)]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaxParameters:
    """Tax-related configuration."""
    filing_status: str = "married_filing_jointly"
    federal_ltcg_rate: float = 0.15
    state_tax_rate: float = 0.055
    marginal_income_tax_rate: float = 0.24
    niit_rate: float = 0.038
    niit_threshold: float = 250_000.0  # MFJ threshold
    estimated_other_magi: float = 360_000.0  # non-investment MAGI


@dataclass
class HELOCParameters:
    """Variable-rate HELOC configuration."""
    draw_schedule: list = field(default_factory=lambda: list(DEFAULT_DRAW_SCHEDULE))
    total_principal: float = 250_000.0
    term_years: int = 15
    spread: float = -0.0021  # rate = prime + spread
    historical_prime_rates: dict = field(default_factory=lambda: dict(HISTORICAL_PRIME_RATES))
    interest_deductible: bool = True

    def get_projected_rates(self) -> dict:
        """
        Project future HELOC rates by reversing the last `term_years` years of
        historical prime rates and applying the spread.
        """
        sorted_years = sorted(self.historical_prime_rates.keys())
        recent = sorted_years[-self.term_years:]
        reversed_primes = [self.historical_prime_rates[y] for y in reversed(recent)]
        start_year = max(sorted_years) + 1
        return {
            start_year + i: prime + self.spread
            for i, prime in enumerate(reversed_primes)
        }


@dataclass
class InvestmentParameters:
    """Investment / liquidation configuration."""
    target_net_proceeds: float = 250_000.0
    cost_basis_ratio: float = 0.60
    assumed_annual_return: float = 0.12
    analysis_horizons: list = field(default_factory=lambda: [10, 15, 20, 25, 30])
    liquidation_schedule: list = field(default_factory=lambda: list(DEFAULT_DRAW_SCHEDULE))


@dataclass
class StrategyResult:
    """Container for a single strategy's analysis output."""
    name: str
    total_cost: float  # nominal dollars spent / lost
    npv_cost: float  # net present value of all costs
    effective_annual_rate: float  # annualized cost as % of principal
    breakdown: Optional[pd.DataFrame] = None
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class HomeImprovementFinanceAnalyzer:
    """Compare staggered liquidation vs. variable-rate HELOC strategies."""

    def __init__(
        self,
        tax: Optional[TaxParameters] = None,
        heloc: Optional[HELOCParameters] = None,
        investment: Optional[InvestmentParameters] = None,
    ):
        self.tax = tax or TaxParameters()
        self.heloc = heloc or HELOCParameters()
        self.inv = investment or InvestmentParameters()

    # -----------------------------------------------------------------------
    # Private utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def _monthly_payment(principal: float, annual_rate: float, months: int) -> float:
        """Standard amortization monthly payment."""
        r = annual_rate / 12
        if r == 0:
            return principal / months
        return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)

    @staticmethod
    def _npv(cashflows: np.ndarray, annual_rate: float) -> float:
        """NPV of annual cashflows (index 0 = year 0)."""
        years = np.arange(len(cashflows))
        return np.sum(cashflows / (1 + annual_rate) ** years)

    @staticmethod
    def _fmt_dollar(v: float) -> str:
        return f"${v:,.0f}"

    @staticmethod
    def _fmt_pct(v: float) -> str:
        return f"{v * 100:.2f}%"

    # -----------------------------------------------------------------------
    # Tax calculations
    # -----------------------------------------------------------------------

    def calculate_capital_gains_tax(
        self,
        gross_proceeds: float,
        cost_basis_ratio: Optional[float] = None,
        include_niit: bool = True,
    ) -> dict:
        """
        Calculate tax on a stock liquidation.

        Returns dict with federal, state, niit, total tax, and net proceeds.
        """
        ratio = cost_basis_ratio if cost_basis_ratio is not None else self.inv.cost_basis_ratio
        cost_basis = gross_proceeds * ratio
        gain = gross_proceeds - cost_basis

        federal = gain * self.tax.federal_ltcg_rate
        state = gain * self.tax.state_tax_rate

        niit = 0.0
        if include_niit:
            magi = self.tax.estimated_other_magi + gain
            excess = max(0, magi - self.tax.niit_threshold)
            niit = self.tax.niit_rate * min(gain, excess)

        total_tax = federal + state + niit
        net_proceeds = gross_proceeds - total_tax

        return {
            "gross_proceeds": gross_proceeds,
            "cost_basis": cost_basis,
            "capital_gain": gain,
            "federal_tax": federal,
            "state_tax": state,
            "niit": niit,
            "total_tax": total_tax,
            "net_proceeds": net_proceeds,
            "effective_tax_rate_on_gain": total_tax / gain if gain > 0 else 0,
        }

    def calculate_grossed_up_liquidation(
        self,
        target_net: Optional[float] = None,
        include_niit: bool = True,
    ) -> dict:
        """
        How much stock to sell to NET the target amount after all taxes.
        """
        target = target_net or self.inv.target_net_proceeds
        gain_fraction = 1 - self.inv.cost_basis_ratio

        if not include_niit:
            combined_rate = self.tax.federal_ltcg_rate + self.tax.state_tax_rate
            gross = target / (1 - gain_fraction * combined_rate)
            return self.calculate_capital_gains_tax(gross, include_niit=False)

        def residual(gross):
            result = self.calculate_capital_gains_tax(gross, include_niit=True)
            return result["net_proceeds"] - target

        lo, hi = target, target * 2
        gross_solved = brentq(residual, lo, hi, xtol=0.01)
        return self.calculate_capital_gains_tax(gross_solved, include_niit=True)

    # -----------------------------------------------------------------------
    # Opportunity cost
    # -----------------------------------------------------------------------

    def calculate_opportunity_cost(
        self,
        amount_liquidated: float,
        years: int,
        annual_return: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Year-by-year lost compound growth from liquidating stock.
        """
        r = annual_return if annual_return is not None else self.inv.assumed_annual_return
        records = []
        for y in range(years + 1):
            value = amount_liquidated * (1 + r) ** y
            lost = value - amount_liquidated
            records.append({"year": y, "portfolio_value": value, "cumulative_lost_growth": lost})
        return pd.DataFrame(records)

    # -----------------------------------------------------------------------
    # HELOC schedule
    # -----------------------------------------------------------------------

    def generate_heloc_schedule(self) -> pd.DataFrame:
        """
        Monthly HELOC simulation with variable rates and staggered draws.

        Draws add to the outstanding balance. The rate changes annually based
        on projected prime rates. When the balance or rate changes, the payment
        is recalculated to amortize the current balance over the remaining months.
        """
        projected_rates = self.heloc.get_projected_rates()
        start_year = min(projected_rates.keys())
        total_months = self.heloc.term_years * 12

        # Build draw lookup: month -> draw amount (draws happen at start of year)
        draw_by_month = {}
        for draw_year, draw_amount in self.heloc.draw_schedule:
            draw_month = draw_year * 12  # month 0, 12, 24, etc.
            draw_by_month[draw_month] = draw_amount

        # Build rate lookup: year index -> annual rate
        rate_by_year_idx = {}
        for cal_year, rate in projected_rates.items():
            rate_by_year_idx[cal_year - start_year] = rate

        balance = 0.0
        current_rate = rate_by_year_idx.get(0, 0.0)
        payment = 0.0
        records = []

        for month in range(total_months):
            year_idx = month // 12
            draw = draw_by_month.get(month, 0.0)

            # Apply draw at start of month
            balance += draw

            # Check for annual rate change
            new_rate = rate_by_year_idx.get(year_idx, current_rate)
            rate_changed = (new_rate != current_rate)
            current_rate = new_rate

            # Recalculate payment when balance changes (draw) or rate changes
            remaining_months = total_months - month
            if (draw > 0 or rate_changed) and balance > 0 and remaining_months > 0:
                payment = self._monthly_payment(balance, current_rate, remaining_months)

            if balance <= 0:
                records.append({
                    "month": month,
                    "payment": 0.0,
                    "principal": 0.0,
                    "interest": 0.0,
                    "tax_savings": 0.0,
                    "remaining_balance": 0.0,
                    "annual_rate": current_rate,
                    "draw_amount": draw,
                })
                continue

            # Last month: adjust payment to zero out balance exactly
            monthly_rate = current_rate / 12
            interest = balance * monthly_rate
            if month == total_months - 1:
                payment = balance + interest

            principal_paid = payment - interest
            balance -= principal_paid

            tax_savings = interest * self.tax.marginal_income_tax_rate if self.heloc.interest_deductible else 0.0

            records.append({
                "month": month,
                "payment": payment,
                "principal": principal_paid,
                "interest": interest,
                "tax_savings": tax_savings,
                "remaining_balance": max(balance, 0.0),
                "annual_rate": current_rate,
                "draw_amount": draw,
            })

        return pd.DataFrame(records)

    def calculate_heloc_total_cost(self) -> dict:
        """Summary HELOC cost metrics."""
        schedule = self.generate_heloc_schedule()
        total_draws = schedule["draw_amount"].sum()
        total_interest = schedule["interest"].sum()
        total_tax_savings = schedule["tax_savings"].sum()
        total_payments = schedule["payment"].sum()
        net_interest = total_interest - total_tax_savings

        # Weighted average rate
        if total_interest > 0:
            schedule_with_interest = schedule[schedule["interest"] > 0]
            weighted_rate = (schedule_with_interest["interest"] * schedule_with_interest["annual_rate"]).sum() / schedule_with_interest["interest"].sum()
        else:
            weighted_rate = 0.0

        return {
            "total_draws": total_draws,
            "total_payments": total_payments,
            "total_interest": total_interest,
            "total_tax_savings": total_tax_savings,
            "net_interest_cost": net_interest,
            "weighted_avg_rate": weighted_rate,
            "schedule": schedule,
        }

    # -----------------------------------------------------------------------
    # Strategy analyzers
    # -----------------------------------------------------------------------

    def analyze_strategy_liquidate(self, horizon_years: int = 20) -> StrategyResult:
        """
        Strategy 1: Staggered stock liquidation on the same schedule as HELOC draws.

        Sells stock at years specified by the liquidation schedule. Taxes are paid
        separately. Dynamic cost basis: as shares appreciate, the cost basis ratio
        relative to current market value shrinks.
        """
        r = self.inv.assumed_annual_return
        base_ratio = self.inv.cost_basis_ratio

        yearly_sales = []  # (year, sale_amount, tax)
        total_tax = 0.0

        for draw_year, amount in self.inv.liquidation_schedule:
            effective_ratio = base_ratio / (1 + r) ** draw_year
            tax_info = self.calculate_capital_gains_tax(
                amount, cost_basis_ratio=effective_ratio, include_niit=True
            )
            tax = tax_info["total_tax"]
            yearly_sales.append((draw_year, amount, tax, tax_info))
            total_tax += tax

        # NPV = sum of (sale + tax) / (1+r)^y
        npv_cost = sum(
            (amount + tax) / (1 + r) ** y
            for y, amount, tax, _ in yearly_sales
        )

        # Nominal lost growth (informational)
        total_lost_growth = 0.0
        for y, amount, _, _ in yearly_sales:
            remaining = horizon_years - y
            if remaining > 0:
                total_lost_growth += amount * ((1 + r) ** remaining - 1)

        total_cost = total_tax + total_lost_growth

        # Breakdown for plotting
        annual_costs = [0.0] * (horizon_years + 1)
        for y, amount, tax, _ in yearly_sales:
            if y <= horizon_years:
                annual_costs[y] = amount + tax
        breakdown = pd.DataFrame({
            "year": range(horizon_years + 1),
            "annual_cost": annual_costs,
        })

        total_sale = sum(amt for _, amt, _, _ in yearly_sales)
        effective_rate = (total_cost / total_sale) ** (1 / horizon_years) - 1 if horizon_years > 0 else 0

        return StrategyResult(
            name="Staggered Liquidation",
            total_cost=total_cost,
            npv_cost=npv_cost,
            effective_annual_rate=effective_rate,
            breakdown=breakdown,
            details={
                "yearly_sales": [
                    {"year": y, "sale_amount": amt, "total_tax": tax,
                     "federal_tax": info["federal_tax"], "state_tax": info["state_tax"],
                     "niit": info["niit"], "capital_gain": info["capital_gain"],
                     "cost_basis": info["cost_basis"]}
                    for y, amt, tax, info in yearly_sales
                ],
                "total_sale": total_sale,
                "total_tax": total_tax,
                "total_lost_growth": total_lost_growth,
                "horizon_years": horizon_years,
            },
        )

    def analyze_strategy_heloc(self, horizon_years: int = 20) -> StrategyResult:
        """
        Strategy 2: Variable-rate HELOC with stock sales to cover payments.

        Uses generate_heloc_schedule(), aggregates annually, models stock sales
        to cover (payment - tax_savings) each year. Same dynamic cost basis
        approach as the liquidation strategy.
        """
        schedule = self.generate_heloc_schedule()
        heloc_cost = self.calculate_heloc_total_cost()

        # Aggregate to annual
        schedule["year"] = schedule["month"] // 12
        annual = schedule.groupby("year").agg(
            payment=("payment", "sum"),
            interest=("interest", "sum"),
            principal=("principal", "sum"),
            tax_savings=("tax_savings", "sum"),
            draw_amount=("draw_amount", "sum"),
        ).reset_index()

        r = self.inv.assumed_annual_return
        base_ratio = self.inv.cost_basis_ratio

        yearly_sales = []  # (year, stock_sold, cap_gains_tax)
        total_stock_sold = 0.0
        total_cap_gains_tax = 0.0

        for _, row in annual.iterrows():
            y = int(row["year"])
            net_cost = row["payment"] - row["tax_savings"]
            if net_cost <= 0:
                yearly_sales.append((y, 0.0, 0.0))
                continue
            stock_to_sell = net_cost
            effective_ratio = base_ratio / (1 + r) ** y
            tax_info = self.calculate_capital_gains_tax(
                stock_to_sell, cost_basis_ratio=effective_ratio, include_niit=True
            )
            cgt = tax_info["total_tax"]
            yearly_sales.append((y, stock_to_sell, cgt))
            total_stock_sold += stock_to_sell
            total_cap_gains_tax += cgt

        # NPV
        npv_cost = sum(
            (sold + cgt) / (1 + r) ** y
            for y, sold, cgt in yearly_sales
        )

        # Total nominal lost growth (informational)
        total_lost_growth = 0.0
        for y, sold, _ in yearly_sales:
            remaining = horizon_years - y
            if remaining > 0:
                total_lost_growth += sold * ((1 + r) ** remaining - 1)

        total_cost = total_cap_gains_tax + total_lost_growth

        # Breakdown for plotting
        annual_costs = [0.0] * (horizon_years + 1)
        for y, sold, cgt in yearly_sales:
            if y <= horizon_years:
                annual_costs[y] = sold + cgt
        breakdown = pd.DataFrame({
            "year": range(horizon_years + 1),
            "annual_cost": annual_costs,
        })

        effective_rate = (total_cost / self.inv.target_net_proceeds) ** (1 / horizon_years) - 1 if horizon_years > 0 else 0

        return StrategyResult(
            name="Variable-Rate HELOC",
            total_cost=total_cost,
            npv_cost=npv_cost,
            effective_annual_rate=effective_rate,
            breakdown=breakdown,
            details={
                "total_draws": heloc_cost["total_draws"],
                "total_payments": heloc_cost["total_payments"],
                "total_interest": heloc_cost["total_interest"],
                "interest_tax_savings": heloc_cost["total_tax_savings"],
                "weighted_avg_rate": heloc_cost["weighted_avg_rate"],
                "total_stock_sold": total_stock_sold,
                "total_cap_gains_tax": total_cap_gains_tax,
                "total_lost_growth": total_lost_growth,
                "horizon_years": horizon_years,
                "yearly_sales": yearly_sales,
            },
        )

    # -----------------------------------------------------------------------
    # Comparison & analysis
    # -----------------------------------------------------------------------

    def compare_all_strategies(self, horizons: Optional[list] = None) -> pd.DataFrame:
        """
        Side-by-side comparison of both strategies across multiple time horizons.
        """
        horizons = horizons or self.inv.analysis_horizons
        rows = []

        for h in horizons:
            liq = self.analyze_strategy_liquidate(horizon_years=h)
            rows.append({
                "strategy": liq.name,
                "horizon_years": h,
                "total_cost": liq.total_cost,
                "npv_cost": liq.npv_cost,
                "effective_annual_rate": liq.effective_annual_rate,
            })

            heloc = self.analyze_strategy_heloc(horizon_years=h)
            rows.append({
                "strategy": heloc.name,
                "horizon_years": h,
                "total_cost": heloc.total_cost,
                "npv_cost": heloc.npv_cost,
                "effective_annual_rate": heloc.effective_annual_rate,
            })

        return pd.DataFrame(rows)

    def sensitivity_analysis(
        self,
        parameter: str = "assumed_annual_return",
        values: Optional[list] = None,
        horizon_years: int = 20,
    ) -> pd.DataFrame:
        """
        Vary one parameter and compare strategy costs.
        """
        if values is None:
            if parameter == "assumed_annual_return":
                values = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
            elif parameter == "cost_basis_ratio":
                values = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
            elif parameter == "state_tax_rate":
                values = [0.0, 0.03, 0.05, 0.055, 0.07, 0.09, 0.10, 0.13]
            elif parameter == "federal_ltcg_rate":
                values = [0.0, 0.10, 0.15, 0.20]
            else:
                raise ValueError(f"Unsupported parameter: {parameter}")

        original_inv_val = getattr(self.inv, parameter, None)
        original_tax_val = getattr(self.tax, parameter, None)

        rows = []
        for v in values:
            if hasattr(self.inv, parameter):
                setattr(self.inv, parameter, v)
            elif hasattr(self.tax, parameter):
                setattr(self.tax, parameter, v)

            liq = self.analyze_strategy_liquidate(horizon_years=horizon_years)
            heloc = self.analyze_strategy_heloc(horizon_years=horizon_years)

            rows.append({
                "parameter": parameter,
                "value": v,
                "liquidation_npv": liq.npv_cost,
                "heloc_npv": heloc.npv_cost,
            })

        if original_inv_val is not None:
            setattr(self.inv, parameter, original_inv_val)
        if original_tax_val is not None:
            setattr(self.tax, parameter, original_tax_val)

        return pd.DataFrame(rows)

    def find_breakeven_return(self, horizon_years: int = 20) -> float:
        """
        Find the portfolio return rate at which liquidation and HELOC have equal
        NPV cost.
        """
        original_return = self.inv.assumed_annual_return

        def cost_diff(r):
            self.inv.assumed_annual_return = r
            liq = self.analyze_strategy_liquidate(horizon_years=horizon_years)
            heloc = self.analyze_strategy_heloc(horizon_years=horizon_years)
            return liq.npv_cost - heloc.npv_cost

        try:
            breakeven = brentq(cost_diff, 0.001, 0.30, xtol=1e-5)
        except ValueError:
            self.inv.assumed_annual_return = original_return
            return float("nan")
        finally:
            self.inv.assumed_annual_return = original_return

        return breakeven

    # -----------------------------------------------------------------------
    # Report generation
    # -----------------------------------------------------------------------

    def generate_report(self) -> str:
        """Comprehensive text report with both strategies, tables, and notes."""
        lines = []
        w = 80
        lines.append("=" * w)
        lines.append("HOME IMPROVEMENT FINANCE ANALYZER — REPORT".center(w))
        lines.append("=" * w)

        # --- Parameters ---
        lines.append("\n--- PARAMETERS ---")
        lines.append(f"  Target net proceeds:       {self._fmt_dollar(self.inv.target_net_proceeds)}")
        lines.append(f"  Filing status:             {self.tax.filing_status}")
        lines.append(f"  Federal LTCG rate:         {self._fmt_pct(self.tax.federal_ltcg_rate)}")
        lines.append(f"  State tax rate:            {self._fmt_pct(self.tax.state_tax_rate)}")
        lines.append(f"  NIIT rate:                 {self._fmt_pct(self.tax.niit_rate)}")
        lines.append(f"  NIIT threshold:            {self._fmt_dollar(self.tax.niit_threshold)}")
        lines.append(f"  Estimated other MAGI:      {self._fmt_dollar(self.tax.estimated_other_magi)}")
        lines.append(f"  Marginal income tax rate:  {self._fmt_pct(self.tax.marginal_income_tax_rate)}")
        lines.append(f"  Cost basis ratio:          {self._fmt_pct(self.inv.cost_basis_ratio)}")
        lines.append(f"  Assumed portfolio return:  {self._fmt_pct(self.inv.assumed_annual_return)}")
        lines.append(f"  Interest deductible:       {'Yes' if self.heloc.interest_deductible else 'No'}")
        lines.append(f"  HELOC spread to prime:     {self.heloc.spread * 100:+.2f}%")
        lines.append(f"  HELOC term:                {self.heloc.term_years} years")

        # Draw schedule
        lines.append(f"\n  Draw schedule:")
        for yr, amt in self.heloc.draw_schedule:
            lines.append(f"    Year {yr}: {self._fmt_dollar(amt)}")

        # Projected rate table
        projected = self.heloc.get_projected_rates()
        lines.append(f"\n  Projected HELOC rates (prime {self.heloc.spread * 100:+.2f}%):")
        rate_items = sorted(projected.items())
        # Print in rows of 5
        for i in range(0, len(rate_items), 5):
            chunk = rate_items[i:i+5]
            parts = [f"    {yr}: {self._fmt_pct(r)}" for yr, r in chunk]
            lines.append("  ".join(parts))

        # --- Strategy 1: Staggered Liquidation ---
        default_h = 20
        lines.append("\n" + "-" * w)
        lines.append("STRATEGY 1: STAGGERED LIQUIDATION")
        lines.append("-" * w)
        liq = self.analyze_strategy_liquidate(horizon_years=default_h)
        d = liq.details
        for sale in d["yearly_sales"]:
            lines.append(f"  Year {sale['year']}:")
            lines.append(f"    Stock sold:              {self._fmt_dollar(sale['sale_amount'])}")
            lines.append(f"    Cost basis:              {self._fmt_dollar(sale['cost_basis'])}")
            lines.append(f"    Capital gain:            {self._fmt_dollar(sale['capital_gain'])}")
            lines.append(f"    Federal tax:             {self._fmt_dollar(sale['federal_tax'])}")
            lines.append(f"    State tax:               {self._fmt_dollar(sale['state_tax'])}")
            lines.append(f"    NIIT:                    {self._fmt_dollar(sale['niit'])}")
            lines.append(f"    Total tax:               {self._fmt_dollar(sale['total_tax'])}")
        lines.append(f"  ---")
        lines.append(f"  Total stock sold:          {self._fmt_dollar(d['total_sale'])}")
        lines.append(f"  Total tax:                 {self._fmt_dollar(d['total_tax'])}")
        lines.append(f"  NPV of cost:               {self._fmt_dollar(liq.npv_cost)}")

        # --- Strategy 2: HELOC ---
        lines.append("\n" + "-" * w)
        lines.append("STRATEGY 2: VARIABLE-RATE HELOC")
        lines.append("-" * w)
        heloc_result = self.analyze_strategy_heloc(horizon_years=default_h)
        d = heloc_result.details
        lines.append(f"  Total draws:               {self._fmt_dollar(d['total_draws'])}")
        lines.append(f"  Total payments:            {self._fmt_dollar(d['total_payments'])}")
        lines.append(f"  Total interest:            {self._fmt_dollar(d['total_interest'])}")
        lines.append(f"  Interest tax savings:      {self._fmt_dollar(d['interest_tax_savings'])}")
        lines.append(f"  Weighted avg rate:         {self._fmt_pct(d['weighted_avg_rate'])}")
        lines.append(f"  Stock sold for payments:   {self._fmt_dollar(d['total_stock_sold'])}")
        lines.append(f"  Cap gains tax on sales:    {self._fmt_dollar(d['total_cap_gains_tax'])}")
        lines.append(f"  NPV of cost:               {self._fmt_dollar(heloc_result.npv_cost)}")

        # Annual HELOC detail
        heloc_cost = self.calculate_heloc_total_cost()
        sched = heloc_cost["schedule"]
        sched["year"] = sched["month"] // 12
        annual_sched = sched.groupby("year").agg(
            payment=("payment", "sum"),
            interest=("interest", "sum"),
            principal=("principal", "sum"),
            tax_savings=("tax_savings", "sum"),
            draw_amount=("draw_amount", "sum"),
        ).reset_index()
        # Get end-of-year rate (last month of each year)
        year_rates = sched.groupby("year")["annual_rate"].last().reset_index()
        annual_sched = annual_sched.merge(year_rates, on="year")

        lines.append(f"\n  Annual HELOC schedule:")
        lines.append(f"  {'Year':>4s}  {'Draw':>10s}  {'Payment':>10s}  {'Interest':>10s}  {'Principal':>10s}  {'Tax Svgs':>10s}  {'Rate':>6s}")
        for _, row in annual_sched.iterrows():
            lines.append(
                f"  {int(row['year']):4d}  "
                f"{self._fmt_dollar(row['draw_amount']):>10s}  "
                f"{self._fmt_dollar(row['payment']):>10s}  "
                f"{self._fmt_dollar(row['interest']):>10s}  "
                f"{self._fmt_dollar(row['principal']):>10s}  "
                f"{self._fmt_dollar(row['tax_savings']):>10s}  "
                f"{self._fmt_pct(row['annual_rate']):>6s}"
            )

        # --- Strategy comparison ---
        lines.append("\n" + "=" * w)
        lines.append("STRATEGY COMPARISON (NPV)")
        lines.append("=" * w)
        comp = self.compare_all_strategies()
        h0 = self.inv.analysis_horizons[0]
        subset = comp[comp["horizon_years"] == h0].sort_values("npv_cost")
        for _, row in subset.iterrows():
            lines.append(
                f"  {row['strategy']:<35s}  "
                f"NPV Cost: {self._fmt_dollar(row['npv_cost']):>12s}"
            )

        diff = liq.npv_cost - heloc_result.npv_cost
        if diff > 0:
            lines.append(f"\n  HELOC saves {self._fmt_dollar(abs(diff))} in NPV vs. liquidation")
        else:
            lines.append(f"\n  Liquidation saves {self._fmt_dollar(abs(diff))} in NPV vs. HELOC")

        # --- Breakeven ---
        lines.append("\n" + "-" * w)
        lines.append("BREAKEVEN ANALYSIS")
        lines.append("-" * w)
        be = self.find_breakeven_return()
        if np.isnan(be):
            lines.append("  No crossover found in 0-30% return range")
        else:
            lines.append(f"  Liquidation cheaper if portfolio return < {self._fmt_pct(be)}")

        # --- Notes ---
        lines.append("\n" + "-" * w)
        lines.append("NOTES")
        lines.append("-" * w)
        lines.append("  - NPV discounted at the assumed portfolio return rate.")
        lines.append("  - HELOC rates projected by reversing last 15 years of prime rates.")
        lines.append("  - NIIT (3.8%) applies to investment income when MAGI exceeds threshold.")
        lines.append("  - HELOC interest deductibility assumes funds used for home improvement")
        lines.append("    on a qualifying residence (TCJA rules).")
        lines.append("  - HELOC payments are modeled as annual stock liquidations,")
        lines.append("    triggering capital gains taxes on each sale.")
        lines.append("=" * w)

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Plotly visualization
    # -----------------------------------------------------------------------

    def generate_plotly_figure(self, horizon_years: int = 20):
        """
        4-subplot visualization:
          1. Bar chart: NPV cost by strategy
          2. Cumulative NPV cost over time
          3. Sensitivity to portfolio return rate
          4. HELOC schedule: stacked bars (principal/interest/tax savings) + rate line
        """
        if not HAS_PLOTLY:
            raise ImportError("plotly is required for visualization. Install with: pip install plotly")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"NPV Cost by Strategy ({horizon_years}yr Horizon)",
                "Cumulative NPV Cost Over Time",
                "Sensitivity to Portfolio Return Rate",
                "HELOC Annual Schedule & Rate",
            ),
            specs=[[{}, {}], [{}, {"secondary_y": True}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
        )

        colors = {
            "liquidation": "#e74c3c",
            "heloc": "#3498db",
        }

        # --- Subplot 1: NPV cost comparison ---
        liq = self.analyze_strategy_liquidate(horizon_years=horizon_years)
        heloc_result = self.analyze_strategy_heloc(horizon_years=horizon_years)

        fig.add_trace(go.Bar(
            x=["Staggered Liquidation", "Variable-Rate HELOC"],
            y=[liq.npv_cost, heloc_result.npv_cost],
            name="NPV Cost",
            marker_color=[colors["liquidation"], colors["heloc"]],
            text=[self._fmt_dollar(liq.npv_cost), self._fmt_dollar(heloc_result.npv_cost)],
            textposition="outside",
        ), row=1, col=1)

        # --- Subplot 2: Cumulative NPV cost over time ---
        years = list(range(horizon_years + 1))
        r = self.inv.assumed_annual_return

        def _cum_npv(annual_costs):
            return np.cumsum([cf / (1 + r) ** y for y, cf in enumerate(annual_costs)]).tolist()

        fig.add_trace(go.Scatter(
            x=years, y=_cum_npv(liq.breakdown["annual_cost"]),
            name="Staggered Liquidation",
            line=dict(color=colors["liquidation"], width=2),
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=years, y=_cum_npv(heloc_result.breakdown["annual_cost"]),
            name="Variable-Rate HELOC",
            line=dict(color=colors["heloc"], width=2),
        ), row=1, col=2)

        # --- Subplot 3: Sensitivity to return rate ---
        sens = self.sensitivity_analysis(
            parameter="assumed_annual_return",
            horizon_years=horizon_years,
        )
        fig.add_trace(go.Scatter(
            x=sens["value"] * 100, y=sens["liquidation_npv"],
            name="Liquidation NPV", line=dict(color=colors["liquidation"], width=2),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=sens["value"] * 100, y=sens["heloc_npv"],
            name="HELOC NPV", line=dict(color=colors["heloc"], width=2),
        ), row=2, col=1)

        # --- Subplot 4: HELOC schedule with rate overlay ---
        heloc_cost = self.calculate_heloc_total_cost()
        sched = heloc_cost["schedule"].copy()
        sched["year"] = sched["month"] // 12
        annual_sched = sched.groupby("year").agg(
            principal=("principal", "sum"),
            interest=("interest", "sum"),
            tax_savings=("tax_savings", "sum"),
        ).reset_index()
        year_rates = sched.groupby("year")["annual_rate"].last().reset_index()
        annual_sched = annual_sched.merge(year_rates, on="year")

        fig.add_trace(go.Bar(
            x=annual_sched["year"], y=annual_sched["principal"],
            name="Principal", marker_color="#3498db", opacity=0.7,
        ), row=2, col=2)
        fig.add_trace(go.Bar(
            x=annual_sched["year"], y=annual_sched["interest"],
            name="Interest", marker_color="#e74c3c", opacity=0.7,
        ), row=2, col=2)
        fig.add_trace(go.Bar(
            x=annual_sched["year"], y=-annual_sched["tax_savings"],
            name="Tax Savings", marker_color="#2ecc71", opacity=0.7,
        ), row=2, col=2)

        # Rate line on secondary y-axis
        fig.add_trace(go.Scatter(
            x=annual_sched["year"], y=annual_sched["annual_rate"] * 100,
            name="HELOC Rate",
            line=dict(color="#f39c12", width=3),
            mode="lines+markers",
        ), row=2, col=2, secondary_y=True)

        fig.update_layout(barmode="relative")

        # --- Layout ---
        fig.update_layout(
            height=900, width=1300,
            title_text="Home Improvement Finance Analysis",
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
            margin=dict(r=180),
        )
        fig.update_yaxes(title_text="NPV Cost ($)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative NPV Cost ($)", row=1, col=2)
        fig.update_xaxes(title_text="Year", row=1, col=2)
        fig.update_yaxes(title_text="NPV Cost ($)", row=2, col=1)
        fig.update_xaxes(title_text="Portfolio Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Annual Amount ($)", row=2, col=2)
        fig.update_yaxes(title_text="Rate (%)", row=2, col=2, secondary_y=True)
        fig.update_xaxes(title_text="Year", row=2, col=2)

        return fig

    # -----------------------------------------------------------------------
    # CLI entry point
    # -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Home Improvement Finance Analyzer: compare stock liquidation vs. variable-rate HELOC"
    )
    parser.add_argument("--return-rate", type=float, default=0.12,
                        help="Assumed annual portfolio return (default: 0.12)")
    parser.add_argument("--basis-ratio", type=float, default=0.60,
                        help="Cost basis as fraction of market value (default: 0.60)")
    parser.add_argument("--state-tax", type=float, default=0.055,
                        help="State capital gains tax rate (default: 0.055)")
    parser.add_argument("--federal-ltcg", type=float, default=0.15,
                        help="Federal LTCG rate (default: 0.15)")
    parser.add_argument("--target", type=float, default=250_000,
                        help="Target net proceeds (default: 250000)")
    parser.add_argument("--other-magi", type=float, default=360_000,
                        help="Estimated non-investment MAGI (default: 360000)")
    parser.add_argument("--horizon", type=int, default=20,
                        help="Primary analysis horizon in years (default: 20)")
    parser.add_argument("--heloc-term", type=int, default=15,
                        help="HELOC term in years (default: 15)")
    parser.add_argument("--heloc-spread", type=float, default=-0.0021,
                        help="HELOC spread to prime rate (default: -0.0021)")
    parser.add_argument("--plot", action="store_true",
                        help="Show interactive Plotly visualization")
    parser.add_argument("--no-niit", action="store_true",
                        help="Exclude NIIT from calculations")

    args = parser.parse_args()

    tax = TaxParameters(
        federal_ltcg_rate=args.federal_ltcg,
        state_tax_rate=args.state_tax,
        estimated_other_magi=args.other_magi,
    )
    heloc = HELOCParameters(
        total_principal=args.target,
        term_years=args.heloc_term,
        spread=args.heloc_spread,
    )
    inv = InvestmentParameters(
        target_net_proceeds=args.target,
        cost_basis_ratio=args.basis_ratio,
        assumed_annual_return=args.return_rate,
    )

    analyzer = HomeImprovementFinanceAnalyzer(tax=tax, heloc=heloc, investment=inv)

    print(analyzer.generate_report())

    if args.plot:
        fig = analyzer.generate_plotly_figure(horizon_years=args.horizon)
        fig.show()


if __name__ == "__main__":
    main()
