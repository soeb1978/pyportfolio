import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linprog


@dataclass
class Employee:
    name: str
    hourly_rate: float           # kr/time
    portfolio_hours: float       # samlet portefølje pr. periode (t)
    teaching_hours: float = 0.0  # obligatorisk undervisning i perioden (t)

@dataclass
class Project:
    name: str
    budget: float               # kr pr. semester


class AllocationError(Exception):
    pass


def allocate_hours_with_nn(
    employees: List[Employee],
    projects: List[Project],
    priorities: Optional[Dict[Tuple[str, str], float]] = None,
    nn_name: str = "NN",
    nn_hourly_rate: float = 600.0,
    nn_max_hours: float = 10_000.0,
) -> Dict[str, np.ndarray]:
    """
    Fordeler timer på projekter for kendte medarbejdere + NN, efter at
    obligatorisk undervisning er trukket ud af porteføljen.

    For hver medarbejder j:
      effektiv_projektportefølje_j = portfolio_hours_j - teaching_hours_j

    LP'en styrer kun de effektive projekttimer.
    """

    n_e = len(employees)
    n_p = len(projects)

    if n_e == 0 or n_p == 0:
        raise ValueError("Der skal være mindst én medarbejder og ét projekt.")

    # Total portefølje og teaching pr. medarbejder
    total_port = np.array([emp.portfolio_hours for emp in employees])
    teaching = np.array([emp.teaching_hours for emp in employees])

    # Effektiv portefølje til projekter
    eff_port = total_port - teaching

    if np.any(eff_port < -1e-6):
        raise AllocationError(
            "En eller flere medarbejdere har teaching_hours > portfolio_hours."
        )

    # Udvidet medarbejderliste (kendte + NN)
    all_names = [emp.name for emp in employees] + [nn_name]
    all_rates = np.array([emp.hourly_rate for emp in employees] + [nn_hourly_rate])

    # Effektiv portefølje til LP’en (kendte + NN)
    all_eff_port = np.concatenate([eff_port, np.array([nn_max_hours])])

    n_e_all = n_e + 1
    n_vars = n_p * n_e_all

    # Objektfunktion: maksimer prioriteret projekttid (minimer -w_ij * h_ij)
    c = np.zeros(n_vars)
    for i, proj in enumerate(projects):
        for j, name in enumerate(all_names):
            k = i * n_e_all + j
            if priorities is not None:
                w = priorities.get((name, proj.name), 1.0)
            else:
                w = 1.0
            c[k] = -w

    # Lighed: projektbudgetter (i kr)
    A_eq = np.zeros((n_p, n_vars))
    b_eq = np.zeros(n_p)

    for i, proj in enumerate(projects):
        b_eq[i] = proj.budget
        for j in range(n_e_all):
            k = i * n_e_all + j
            A_eq[i, k] = all_rates[j]

    # Ulighed: timeloft pr. medarbejder (effektiv portefølje)
    A_ub = np.zeros((n_e_all, n_vars))
    b_ub = all_eff_port.copy()

    for j in range(n_e_all):
        for i in range(n_p):
            k = i * n_e_all + j
            A_ub[j, k] = 1.0

    bounds = [(0, None)] * n_vars

    res = linprog(
        c=c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method="highs"
    )

    if not res.success:
        raise AllocationError(f"Løsning ikke mulig: {res.message}")

    x = res.x
    hours = x.reshape((n_p, n_e_all))  # projekttimer (P × (E+1))

    # Projekttimer pr. medarbejder (inkl. NN)
    proj_hours_per_emp = hours.sum(axis=0)  # (E+1,)

    # Centertid = total_port - teaching - projekttimer
    centertime = np.maximum(
        total_port - teaching - proj_hours_per_emp[:n_e],
        0.0
    )

    return {
        "hours_projects": hours,
        "centertime": centertime,               # (E,)
        "names_employees": all_names,          # (E+1,)
        "rates": all_rates,                    # (E+1,)
        "effective_portfolio_all": all_eff_port,  # (E+1,)
        "total_portfolio": total_port,         # (E,)
        "teaching": teaching,                  # (E,)
    }

def _short_label(name: str) -> str:
    """
    Trækker kortkode ud af navnet:
    'Søren Erbs Poulsen (SOEB)' -> 'SOEB'
    Bruges kun internt hvis du senere vil have korte labels.
    """
    if "(" in name and ")" in name:
        try:
            return name.split("(")[-1].split(")")[0].strip()
        except Exception:
            return name
    return name


def print_allocation_report(
    employees: List[Employee],
    projects: List[Project],
    allocation_result: Dict[str, np.ndarray],
) -> None:
    """
    Kompakt og læsbar rapport:

      1) Medarbejderoversigt (fulde navne, løn, portefølje, undervisning, projekttimer, centertid)
      2) Projektoversigt (budgetter)
      3) Allokering (pivot):
         - rækker: medarbejdere (inkl. NN)
         - kolonner: projekter
         - sidste kolonne: sum projekttimer pr. medarbejder
    """

    hours = allocation_result["hours_projects"]                # (P, E+1)
    centertime = allocation_result["centertime"]               # (E,)
    all_names = allocation_result["names_employees"]           # (E+1,)
    all_rates = allocation_result["rates"]                     # (E+1,)
    total_port = allocation_result["total_portfolio"]          # (E,)
    teaching = allocation_result["teaching"]                   # (E,)

    n_p, n_e_all = hours.shape
    n_e = len(employees)

    # Projekttimer pr. medarbejder (inkl. NN)
    proj_hours_per_emp = hours.sum(axis=0)           # (E+1,)

    # ---------------- 1) Medarbejdere ----------------
    print("\n=== MEDARBEJDERE (INPUT + RESULTAT) ===\n")
    print("{:<35}{:>10}{:>12}{:>12}{:>12}{:>12}".format(
        "Medarbejder", "Timepris", "Portef.[t]", "Underv.[t]",
        "Proj.[t]", "Center[t]"
    ))

    for j, emp in enumerate(employees):
        port = total_port[j]
        teach = teaching[j]
        proj_h = proj_hours_per_emp[j]
        center_h = centertime[j]

        print("{:<35}{:>10.0f}{:>12.1f}{:>12.1f}{:>12.1f}{:>12.1f}".format(
            emp.name[:35],
            emp.hourly_rate,
            port,
            teach,
            proj_h,
            center_h,
        ))

    # NN-medarbejder
    nn_name = all_names[-1]
    nn_rate = all_rates[-1]
    nn_proj_h = proj_hours_per_emp[-1]

    print("{:<35}{:>10.0f}{:>12}{:>12}{:>12.1f}{:>12}".format(
        nn_name[:35],
        nn_rate,
        "-",
        "-",
        nn_proj_h,
        "-",
    ))

    # ---------------- 2) Projekter ----------------
    print("\n=== PROJEKTER (INPUT) ===\n")
    print("{:<20}{:>15}".format("Projekt", "Budget [kr]"))
    for proj in projects:
        print("{:<20}{:>15.0f}".format(proj.name, proj.budget))

    # ---------------- 3) Allokering (pivot) ----------------
    print("\n=== ALLOKERING – MEDARBEJDERE × PROJEKTER (HELE TIMER) ===\n")

    proj_names = [p.name for p in projects]

    # Header: medarbejder + én kolonne pr. projekt + sum
    print("{:<35}".format("Medarbejder"), end="")
    for pname in proj_names:
        print("{:>12}".format(pname[:11]), end="")
    print("{:>12}".format("Sum proj."))
    print()

    # Pivot: rows = medarbejdere, cols = projekter
    hours_T = hours.T   # (E+1, P)

    # Kendte medarbejdere
    for j, emp in enumerate(employees):
        row = hours_T[j, :]          # (P,)
        row_sum = proj_hours_per_emp[j]
        print("{:<35}".format(emp.name[:35]), end="")
        for i in range(n_p):
            print("{:>12.0f}".format(round(row[i])), end="")
        print("{:>12.0f}".format(round(row_sum)))

    # NN-række
    nn_row = hours_T[-1, :]
    print("{:<35}".format(nn_name[:35]), end="")
    for i in range(n_p):
        print("{:>12.0f}".format(round(nn_row[i])), end="")
    print("{:>12.0f}".format(round(nn_proj_h)))

    print()  # ekstra linjeskift til sidst
