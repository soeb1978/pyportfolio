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

def print_allocation_report(
    employees: List[Employee],
    projects: List[Project],
    allocation_result: Dict[str, np.ndarray],
) -> None:
    hours = allocation_result["hours_projects"]           # (P, E+1)
    centertime = allocation_result["centertime"]          # (E,)
    all_names = allocation_result["names_employees"]      # (E+1,)
    all_rates = allocation_result["rates"]                # (E+1,)
    all_eff_port = allocation_result["effective_portfolio_all"]  # (E+1,)
    total_port = allocation_result["total_portfolio"]     # (E,)
    teaching = allocation_result["teaching"]              # (E,)

    n_p, n_e_all = hours.shape
    n_e = len(employees)

    # --------- 1) Medarbejderinput ---------
    print("\n=== Medarbejdere (input) ===\n")
    print("{:<20}{:>15}{:>18}{:>18}{:>20}".format(
        "Medarbejder", "Timepris", "Portefølje", "Undervisning", "Til projekter"
    ))
    for j, emp in enumerate(employees):
        eff = total_port[j] - teaching[j]
        print("{:<20}{:>15.2f}{:>18.2f}{:>18.2f}{:>20.2f}".format(
            emp.name,
            emp.hourly_rate,
            total_port[j],
            teaching[j],
            eff
        ))

    # NN
    nn_name = all_names[-1]
    nn_rate = all_rates[-1]
    nn_eff = all_eff_port[-1]
    print("{:<20}{:>15.2f}{:>18}{:>18}{:>20.2f}  (NN)".format(
        nn_name, nn_rate, "-", "-", nn_eff
    ))

    # --------- 2) Projekter ---------
    print("\n=== Projekter (input) ===\n")
    print("{:<20}{:>20}".format("Projekt", "Budget [kr]"))
    for proj in projects:
        print("{:<20}{:>20.2f}".format(proj.name, proj.budget))

    # --------- 3) Allokering pr. projekt ---------
    print("\n=== Allokering pr. projekt (timer og økonomi) ===\n")

    header = ["Projekt"] + [f"{name} [t]" for name in all_names] + [
        "Sum timer", "Sum løn", "Budget", "Afvigelse"
    ]
    print("{:<20}".format(header[0]), end="")
    for h in header[1:]:
        print("{:>15}".format(h), end="")
    print()

    for i, proj in enumerate(projects):
        row_hours = hours[i, :]
        row_costs = row_hours * all_rates
        sum_hours = float(row_hours.sum())
        sum_cost = float(row_costs.sum())
        budget = proj.budget
        diff = sum_cost - budget

        print("{:<20}".format(proj.name), end="")
        for j in range(n_e_all):
            print("{:>15.2f}".format(row_hours[j]), end="")
        print("{:>15.2f}{:>15.2f}{:>15.2f}{:>15.2f}".format(
            sum_hours, sum_cost, budget, diff
        ))

    # --------- 4) Medarbejderporteføljer ---------
    print("\n=== Medarbejderporteføljer (projekter, undervisning, centertid) ===\n")
    print("{:<20}{:>15}{:>15}{:>15}{:>15}{:>15}".format(
        "Medarbejder", "Portefølje", "Undervisning", "Proj.timer",
        "Centertid", "Proj.andel"
    ))

    proj_hours_per_emp = hours.sum(axis=0)  # (E+1,)

    # Kendte medarbejdere
    for j, emp in enumerate(employees):
        port = total_port[j]
        teach = teaching[j]
        eff = port - teach
        proj_h = proj_hours_per_emp[j]
        center_h = centertime[j]
        proj_share = (proj_h / eff * 100.0) if eff > 0 else 0.0

        print("{:<20}{:>15.2f}{:>15.2f}{:>15.2f}{:>15.2f}{:>15.1f}".format(
            emp.name, port, teach, proj_h, center_h, proj_share
        ))

    # NN
    nn_proj_h = proj_hours_per_emp[-1]
    nn_eff = all_eff_port[-1]
    nn_share = (nn_proj_h / nn_eff * 100.0) if nn_eff > 0 else 0.0

    print("{:<20}{:>15}{:>15}{:>15.2f}{:>15}{:>15.1f}".format(
        nn_name, "-", "-", nn_proj_h, "-", nn_share
    ))
