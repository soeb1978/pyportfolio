import numpy as np
import pandas as pd
from typing import List, Tuple

from portfolio_model import (
    Employee, Project,
    allocate_hours,
    print_allocation_report,
    export_portfolio_to_excel,
    AllocationError,
)


def load_rates_and_budgets_from_excel(
    path: str,
    sheet_name: str,
    employees: List[Employee],
    projects: List[Project],
    budget_row_label: str = "Project budget [DKR]",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Indlæser timesatser og projektbudgetter fra en Excel-fil med struktur:

      Ark: sheet_name
      Kolonne 'Medarbejder' = rækker (medarbejdere, inkl. 'NN (ufordelt)')
      Øvrige kolonner = projekter (navne som i Project.name)

      Række 'Project budget [DKR]' bruges som projektbudgetter [kr].

    Returnerer:
      - rate_matrix  : np.ndarray med form (n_projects, n_employees)
      - budgets_arr  : np.ndarray med form (n_projects,)
    """

    df = pd.read_excel(path, sheet_name=sheet_name)
    if "Medarbejder" not in df.columns:
        raise ValueError("Excel-arket skal have en kolonne 'Medarbejder'.")

    df = df.set_index("Medarbejder")

    proj_names = [p.name for p in projects]
    emp_names = [e.name for e in employees]

    # Budgetrække
    if budget_row_label not in df.index:
        raise ValueError(f"Kunne ikke finde budgetrækken '{budget_row_label}' i Excel-arket.")

    budgets_series = df.loc[budget_row_label, proj_names]
    budgets_arr = budgets_series.to_numpy(dtype=float)

    # Fjern KUN budgetrækken – alle medarbejdere, inkl. 'NN (ufordelt)', bliver i datamatricen
    df_rates = df.drop(index=[budget_row_label])

    # Tjek at alle medarbejdere i Python findes i Excel
    missing_emps = set(emp_names) - set(df_rates.index)
    if missing_emps:
        raise ValueError(f"Følgende medarbejdere mangler i Excel-arket: {missing_emps}")

    missing_projects = set(proj_names) - set(df_rates.columns)
    if missing_projects:
        raise ValueError(f"Følgende projekter mangler i Excel-arket: {missing_projects}")

    # Sortér i samme rækkefølge som employees og projects
    df_rates = df_rates.loc[emp_names, proj_names]

    # rows = employees, cols = projekter -> transpose til (n_projects, n_employees)
    rate_matrix = df_rates.to_numpy(dtype=float).T

    return rate_matrix, budgets_arr


def main():
    semester = "F2026"

    employees = [
        Employee("Søren Erbs Poulsen (SOEB)", hourly_rate=656.64, portfolio_hours=657.5, teaching_hours=0),
        Employee("Mathias Larsen (MATL)",   hourly_rate=531.36, portfolio_hours=657.5, teaching_hours=450.0),   #150 på Ronja, Torben 300
        Employee("Rune Kier Nielsen (RUNI)", hourly_rate=560.14, portfolio_hours=1315/2-480, teaching_hours=0),
        Employee("Kristoffer Bested Nielsen (KRI)",  hourly_rate=499.35, portfolio_hours=657.5, teaching_hours=547.0),   
        Employee("Søren Andersen (SSSA)", hourly_rate=544.32, portfolio_hours=657.5, teaching_hours=500.0),           
        Employee("Karl Woldum Tordrup (KART)", hourly_rate=600.31, portfolio_hours=100, teaching_hours=0),
        Employee("Marton Major (MMAJ)", hourly_rate=529.01, portfolio_hours=657.5, teaching_hours=0),

        # NN – meget stor portefølje, ingen undervisning
        Employee("NN (ufordelt)",            hourly_rate=535.90, portfolio_hours=10_000.0, teaching_hours=0),
    ]

    # Projekter – budget overskrives fra Excel
    projects = [
        Project("The Change",      budget=0.0),
        Project("LEG-DHC",         budget=0.0),
        Project("LTDE-repBC",      budget=0.0),
        Project("COOLGEOHEAT II",  budget=0.0),
        Project("HEATCODE",        budget=0.0),
    ]

    # Indlæs timesatser og budgetter fra filen
    rate_matrix, budgets_arr = load_rates_and_budgets_from_excel(
        path="time_satser_F2026.xlsx",          # tilpas evt. sti
        sheet_name="Timesatser_budget",
        employees=employees,
        projects=projects,
    )

    # Opdatér projektbudgetter fra Excel
    for proj, b in zip(projects, budgets_arr):
        proj.budget = float(b)

    # Prioriteter – eksempelvis høj prioritet på COOLGEOHEAT II for visse personer
    priorities = {
        ("Søren Erbs Poulsen (SOEB)", "LTDE-repBC"): 4.0,
        ("Søren Erbs Poulsen (SOEB)", "COOLGEOHEAT II"): 3.0,
        ("Søren Erbs Poulsen (SOEB)", "The Change"): 2.0,
        ("Kristoffer Bested Nielsen (KRI)", "COOLGEOHEAT II"): 3.0,
        ("Marton Major (MMAJ)", "LEG-DHC"): 3.0,
        ("Karl Woldum Tordrup (KART)", "The Change"): 3.0,
        ("Rune Kier Nielsen (RUNI)", "LTDE-repBC"): 2.9,
        ("Rune Kier Nielsen (RUNI)", "COOLGEOHEAT II"): 2.0,
        ("Søren Andersen (SSSA)", "HEATCODE"): 3.0,
        ("Mathias Larsen (MATL)", "COOLGEOHEAT II"): 3.0,
    }

    try:
        result = allocate_hours(
            employees=employees,
            projects=projects,
            priorities=priorities,
            project_hourly_rates=rate_matrix,
        )
    except AllocationError as e:
        print("Porteføljeberegning fejlede:")
        print(e)
        return

    print_allocation_report(employees, projects, result)

    export_portfolio_to_excel(
        employees=employees,
        projects=projects,
        allocation_result=result,
        semester_label=semester,
        source_excel_path="time_satser_F2026.xlsx",   # inputfil med timesatser + budgetter
        source_sheet_name="Timesatser_budget",        # arket der skal kopieres
    )
    print(f"Porteføljen er gemt i 'Portefølje_{semester}.xlsx'.")


if __name__ == "__main__":
    main()
