import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linprog
import pandas as pd


@dataclass
class Employee:
    name: str
    portfolio_hours: float       # samlet portefølje pr. periode (t)
    teaching_hours: float = 0.0  # obligatorisk undervisning i perioden (t)


@dataclass
class Project:
    name: str
    budget: float                # kr pr. periode (lønbudget)


class AllocationError(Exception):
    """Kastes hvis der ikke kan findes en gennemførlig løsning."""
    pass


def allocate_hours(
    employees: List[Employee],
    projects: List[Project],
    priorities: Optional[Dict[Tuple[str, str], float]] = None,
    project_hourly_rates: Optional[np.ndarray] = None,
    preallocated_hours: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Fordeler projekttimer på ALLE medarbejdere (inkl. 'NN (ufordelt)'),
    efter at obligatorisk undervisning er trukket ud af porteføljen.

    Understøtter preallokering pr. projekt/medarbejder:

        h_ij = preallocated_hours_ij + x_ij,  x_ij >= 0

    hvor LP'en bestemmer x_ij (ekstra timer) givet restbudget og restkapacitet.

    PARAMETRE
    ---------
    employees : liste af Employee
    projects  : liste af Project
    priorities: dict med nøgler (medarbejdernavn, projektnavn) og værdier (weights)
                bruges til at vægte hvilke projekt/medarbejder-kombinationer, der
                foretrækkes. Default 1.0 hvis ikke angivet.
    project_hourly_rates : np.ndarray med form (n_projects, n_employees)
                element (i,j) er timeprisen [kr/t] for projekt i og medarbejder j.
                Denne matrix er OBLIGATORISK – satser fra Excel.
    preallocated_hours : np.ndarray med form (n_projects, n_employees) eller None
                preallokerede timer på hver (projekt, medarbejder). Hvis None ⇒ 0.

    RETURNERER
    ----------
    Dict med nøgler:
      - "hours_projects":   (n_projects, n_employees) totale projekttimer
                            (preallokering + ekstra)
      - "centertime":       (n_employees,) centertid pr. medarbejder
      - "names_employees":  liste af medarbejdernavne (længde n_employees)
      - "total_portfolio":  (n_employees,) total portefølje pr. medarbejder
      - "teaching":         (n_employees,) undervisningstimer pr. medarbejder
      - "rate_matrix":      (n_projects, n_employees) satser fra Excel
      - "preallocated_hours": (n_projects, n_employees) preallokering
      - "extra_hours":        (n_projects, n_employees) ekstra timer fra LP
    """

    n_e = len(employees)
    n_p = len(projects)

    if n_e == 0 or n_p == 0:
        raise ValueError("Der skal være mindst én medarbejder og ét projekt.")

    if project_hourly_rates is None:
        raise ValueError(
            "project_hourly_rates skal angives og komme fra Excel (n_projects × n_employees)."
        )

    # Total portefølje og undervisning pr. medarbejder
    total_port = np.array([emp.portfolio_hours for emp in employees], dtype=float)
    teaching = np.array([emp.teaching_hours for emp in employees], dtype=float)

    # Effektiv portefølje til projekter
    eff_port = total_port - teaching
    if np.any(eff_port < -1e-6):
        raise AllocationError(
            "En eller flere medarbejdere har teaching_hours > portfolio_hours."
        )

    names = [emp.name for emp in employees]

    # Rate-matrix pr. projekt og medarbejder (P × E)
    rate_matrix = np.array(project_hourly_rates, dtype=float)
    if rate_matrix.shape != (n_p, n_e):
        raise ValueError(
            f"project_hourly_rates skal have form (n_projects, n_employees) = ({n_p}, {n_e}), "
            f"men har {rate_matrix.shape}"
        )

    # Preallokerede timer (P × E)
    if preallocated_hours is None:
        prealloc = np.zeros((n_p, n_e), dtype=float)
    else:
        prealloc = np.array(preallocated_hours, dtype=float)
        if prealloc.shape != (n_p, n_e):
            raise ValueError(
                f"preallocated_hours skal have form (n_projects, n_employees) = ({n_p}, {n_e}), "
                f"men har {prealloc.shape}"
            )

    # ---- Tjek og beregn restbudget & restkapacitet ----
    # Forbrug af budget pga. preallokering
    prealloc_costs_per_proj = (prealloc * rate_matrix).sum(axis=1)  # (P,)
    budgets = np.array([p.budget for p in projects], dtype=float)
    budget_residual = budgets - prealloc_costs_per_proj

    if np.any(budget_residual < -1e-6):
        raise AllocationError(
            "Preallokerede timer overstiger projektbudgettet for mindst ét projekt."
        )

    budget_residual = np.maximum(budget_residual, 0.0)

    # Forbrug af effektiv portefølje pga. preallokering
    prealloc_hours_per_emp = prealloc.sum(axis=0)  # (E,)
    eff_port_residual = eff_port - prealloc_hours_per_emp

    if np.any(eff_port_residual < -1e-6):
        raise AllocationError(
            "Preallokerede timer overstiger den tilgængelige projektportefølje "
            "for mindst én medarbejder."
        )

    eff_port_residual = np.maximum(eff_port_residual, 0.0)

    # ---- LP på ekstra timer x_ij ----
    n_vars = n_p * n_e

    # Objektfunktion: maksimer prioriteret projekttid (minimer -w_ij * x_ij)
    c = np.zeros(n_vars)
    for i, proj in enumerate(projects):
        for j, name in enumerate(names):
            k = i * n_e + j
            w = priorities.get((name, proj.name), 1.0) if priorities else 1.0
            c[k] = -w

    # Lighed: restbudget pr. projekt (i kr)
    A_eq = np.zeros((n_p, n_vars))
    b_eq = budget_residual.copy()

    for i in range(n_p):
        for j in range(n_e):
            k = i * n_e + j
            A_eq[i, k] = rate_matrix[i, j]   # kr/time

    # Ulighed: restkapacitet pr. medarbejder (effektiv portefølje)
    A_ub = np.zeros((n_e, n_vars))
    b_ub = eff_port_residual.copy()

    for j in range(n_e):
        for i in range(n_p):
            k = i * n_e + j
            A_ub[j, k] = 1.0

    bounds = [(0, None)] * n_vars

    res = linprog(
        c=c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        raise AllocationError(f"Løsning ikke mulig: {res.message}")

    x = res.x
    extra_hours = x.reshape((n_p, n_e))  # (P × E)

    # Totale projekttimer = preallokerede + ekstra
    hours_total = prealloc + extra_hours

    # Projekttimer pr. medarbejder
    proj_hours_per_emp = hours_total.sum(axis=0)  # (E,)

    # Centertid = total_port - teaching - projekttimer
    centertime = np.maximum(
        total_port - teaching - proj_hours_per_emp,
        0.0,
    )

    return {
        "hours_projects": hours_total,      # (P, E)
        "centertime": centertime,           # (E,)
        "names_employees": names,           # (E,)
        "total_portfolio": total_port,      # (E,)
        "teaching": teaching,               # (E,)
        "rate_matrix": rate_matrix,         # (P, E)
        "preallocated_hours": prealloc,     # (P, E)
        "extra_hours": extra_hours,         # (P, E)
    }


def print_allocation_report(
    employees: List[Employee],
    projects: List[Project],
    allocation_result: Dict[str, np.ndarray],
) -> None:
    """
    Konsolrapport:

      1) Allokeringsoversigt (pivot):
         - rækker: medarbejdere (inkl. 'NN (ufordelt)')
         - kolonner: projekter (timer, afrundet)
         - ekstra kolonner: Centertid, Sum tid, Portefølje, Ekstern (undervisning)
      2) Kontrol: projektbudget vs. beregnet omkostning (på afrundede timer)
    """

    hours = allocation_result["hours_projects"]                # (P, E)
    centertime = allocation_result["centertime"]               # (E,)
    names = allocation_result["names_employees"]               # (E,)
    total_port = allocation_result["total_portfolio"]          # (E,)
    teaching = allocation_result["teaching"]                   # (E,)
    rate_matrix = allocation_result["rate_matrix"]             # (P, E)

    n_p, n_e = hours.shape

    proj_hours_per_emp = hours.sum(axis=0)           # (E,)

    # Brug afrundede timer (hele timer) til omkostningsberegning og visning
    hours_rounded = np.round(hours)                    # (P, E)
    project_costs = (hours_rounded * rate_matrix).sum(axis=1)  # (P,)
    total_cost_all = float(project_costs.sum())

    # ---------------- ALLOKERING (pivot) ----------------
    print("\n=== ALLOKERING – MEDARBEJDERE × PROJEKTER (HELE TIMER) ===\n")

    proj_names = [p.name for p in projects]

    # Header: medarbejder + projekter + Centertid + Sum tid + Portefølje + Ekstern
    print("{:<35}".format("Medarbejder"), end="")
    for pname in proj_names:
        print("{:>15}".format(pname[:11]), end="")
    print("{:>15}{:>15}{:>15}{:>15}".format("Centertid", "Sum tid", "Portefølje", "Ekstern"))

    hours_T = hours_rounded.T   # (E, P) – visning i hele timer

    for j, emp in enumerate(employees):
        row = hours_T[j, :]                # (P,)
        proj_sum = row.sum()               # sum projekttimer (afrundede)
        center_h = centertime[j]           # centertid
        port = total_port[j]               # samlet portefølje
        teach = teaching[j]                # undervisning

        print("{:<35}".format(emp.name[:35]), end="")
        for i in range(n_p):
            print("{:>15.0f}".format(row[i]), end="")

        if emp.name == "NN (ufordelt)":
            # For NN: skjul alle kolonner fra og med Centertid
            cent_str = "-"
            sum_tid_str = "-"
            port_str = "-"
            teach_str = "-"
        else:
            sum_tid = proj_sum + center_h
            cent_str = f"{center_h:.0f}"
            sum_tid_str = f"{sum_tid:.0f}"
            port_str = f"{port:.0f}"
            teach_str = f"{teach:.0f}"

        print("{:>15}{:>15}{:>15}{:>15}".format(
            cent_str,
            sum_tid_str,
            port_str,
            teach_str,
        ))

    # ---------------- KONTROL: budget vs. omkostning ----------------
    print("\n=== KONTROL: BUDGET VS. BEREGNET OMKOSTNING [kr] (PÅ AFRUNDEDE TIMER) ===\n")
    print("{:<20}{:>15}{:>15}{:>15}".format(
        "Projekt", "Budget", "Omkostning", "Afvigelse",
    ))
    for i, proj in enumerate(projects):
        budget = proj.budget
        cost = project_costs[i]
        diff = cost - budget
        print("{:<20}{:>15.0f}{:>15.0f}{:>15.0f}".format(
            proj.name[:20],
            budget,
            cost,
            diff,
        ))

    total_budget = sum(p.budget for p in projects)
    total_diff = total_cost_all - total_budget

    print("\n{:<20}{:>15.0f}{:>15.0f}{:>15.0f}".format(
        "I alt",
        total_budget,
        total_cost_all,
        total_diff,
    ))
    print()


def portfolio_to_dataframe(
    employees: List[Employee],
    projects: List[Project],
    allocation_result: Dict[str, np.ndarray],
    semester_label: str,
) -> pd.DataFrame:
    """
    Returnerer en DataFrame med porteføljen for et givent semester.

    Rækker:
      - én række pr. medarbejder (inkl. 'NN (ufordelt)')
      - én række for '*** Projektbudget [kr]'
      - én række for '*** Sum projektløn [kr]'

    Kolonner:
      - én kolonne pr. projekt (timer for medarbejdere, kr for de to bundrækker)
      - 'Sum projekter [t]'
      - 'Undervisning [t]'
      - 'Centertid [t]'
      - 'Portefølje total [t]'
      - 'Periode'

    Alle timekolonner afrundes til heltal i output (til brug i Excel).
    """

    hours = allocation_result["hours_projects"]          # (P, E)
    centertime = allocation_result["centertime"]         # (E,)
    names = allocation_result["names_employees"]         # (E,)
    total_port = allocation_result["total_portfolio"]    # (E,)
    teaching = allocation_result["teaching"]             # (E,)
    rate_matrix = allocation_result["rate_matrix"]       # (P, E)

    proj_names = [p.name for p in projects]

    # Basis: medarbejdere × projekter (timer)
    df = pd.DataFrame(
        hours.T,               # (E, P)
        index=names,
        columns=proj_names,
    )

    # Sum projekttimer pr. medarbejder
    df["Sum projekter [t]"] = df[proj_names].sum(axis=1)

    df["Undervisning [t]"] = teaching
    df["Centertid [t]"] = centertime
    df["Portefølje total [t]"] = total_port

    df["Periode"] = semester_label

    # Skjul centertid i Excel for NN (ufordelt)
    nn_label = "NN (ufordelt)"
    if nn_label in df.index:
        df.loc[nn_label, "Centertid [t]"] = np.nan

    # 1) Projektbudgetter [kr]
    budget_row_label = "*** Projektbudget [kr]"
    project_budgets = np.array([p.budget for p in projects], dtype=float)  # (P,)
    df.loc[budget_row_label, proj_names] = np.round(project_budgets, 0)
    df.loc[budget_row_label, "Sum projekter [t]"] = np.nan
    df.loc[budget_row_label, "Undervisning [t]"] = np.nan
    df.loc[budget_row_label, "Centertid [t]"] = np.nan
    df.loc[budget_row_label, "Portefølje total [t]"] = np.nan
    df.loc[budget_row_label, "Periode"] = semester_label

    # 2) Samlet projektløn [kr] pr. projekt baseret på afrundede timer
    hours_rounded = np.round(hours)                         # (P, E)
    project_costs = (hours_rounded * rate_matrix).sum(axis=1)  # (P,)
    cost_row_label = "*** Sum projektløn [kr]"
    df.loc[cost_row_label, proj_names] = np.round(project_costs, 0)
    df.loc[cost_row_label, "Sum projekter [t]"] = np.nan
    df.loc[cost_row_label, "Undervisning [t]"] = np.nan
    df.loc[cost_row_label, "Centertid [t]"] = np.nan
    df.loc[cost_row_label, "Portefølje total [t]"] = np.nan
    df.loc[cost_row_label, "Periode"] = semester_label

    # Afrund alle time-relaterede kolonner til heltal
    hour_cols = proj_names + [
        "Sum projekter [t]",
        "Undervisning [t]",
        "Centertid [t]",
        "Portefølje total [t]",
    ]
    df[hour_cols] = df[hour_cols].round(0)

    df.index.name = "Medarbejder"
    return df


def export_portfolio_to_excel(
    employees: List[Employee],
    projects: List[Project],
    allocation_result: Dict[str, np.ndarray],
    semester_label: str,
    filename: Optional[str] = None,
    source_excel_path: Optional[str] = None,
    source_sheet_name: str = "Timesatser_budget",
) -> None:
    """
    Gemmer porteføljen i et Excel-ark.

    Hvis filename ikke angives, bruges:
        f"Portefølje_{semester_label}.xlsx"

    Hvis source_excel_path angives, kopieres arket `source_sheet_name`
    fra denne fil med ind som et ekstra ark i resultatfilen, så input
    (timesatser + budgetter) og output er samlet ét sted.
    """

    if filename is None:
        filename = f"Portefølje_{semester_label}.xlsx"

    df_port = portfolio_to_dataframe(
        employees=employees,
        projects=projects,
        allocation_result=allocation_result,
        semester_label=semester_label,
    )

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        # Output-ark
        df_port.to_excel(
            writer,
            sheet_name=f"Portefølje_{semester_label}",
            index=True,
        )

        # Input-ark kopieret fra kildefilen (valgfrit)
        if source_excel_path is not None:
            df_input = pd.read_excel(source_excel_path, sheet_name=source_sheet_name)
            df_input.to_excel(
                writer,
                sheet_name=source_sheet_name,
                index=False,
            )
