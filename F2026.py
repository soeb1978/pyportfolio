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
        raise ValueError(f"Følgende medarbejdere mangler i Excel-arket (timesatser): {missing_emps}")

    missing_projects = set(proj_names) - set(df_rates.columns)
    if missing_projects:
        raise ValueError(f"Følgende projekter mangler i Excel-arket (timesatser): {missing_projects}")

    # Sortér i samme rækkefølge som employees og projects
    df_rates = df_rates.loc[emp_names, proj_names]

    # rows = employees, cols = projekter -> transpose til (n_projects, n_employees)
    rate_matrix = df_rates.to_numpy(dtype=float).T

    return rate_matrix, budgets_arr


def load_preallocated_hours_from_excel(
    path: str,
    sheet_name: str,
    employees: List[Employee],
    projects: List[Project],
) -> np.ndarray:
    """
    Indlæser preallokerede timer fra et Excel-ark med struktur:

      Ark: sheet_name (fx 'Preallokering')
      Kolonne 'Medarbejder' = rækker (medarbejdere, inkl. 'NN (ufordelt)')
      Øvrige kolonner = projekter (navne som i Project.name)

    Returnerer:
      prealloc_matrix : np.ndarray med form (n_projects, n_employees)
        element (i,j) = preallokerede timer for projekt i og medarbejder j.
    """

    df = pd.read_excel(path, sheet_name=sheet_name)
    if "Medarbejder" not in df.columns:
        raise ValueError("Preallokerings-arket skal have en kolonne 'Medarbejder'.")

    df = df.set_index("Medarbejder")

    proj_names = [p.name for p in projects]
    emp_names = [e.name for e in employees]

    missing_emps = set(emp_names) - set(df.index)
    if missing_emps:
        raise ValueError(f"Følgende medarbejdere mangler i preallokeringsarket: {missing_emps}")

    missing_projects = set(proj_names) - set(df.columns)
    if missing_projects:
        raise ValueError(f"Følgende projekter mangler i preallokeringsarket: {missing_projects}")

    df_sub = df.loc[emp_names, proj_names].fillna(0.0)

    # rows = employees, cols = projekter -> transpose til (n_projects, n_employees)
    prealloc_matrix = df_sub.to_numpy(dtype=float).T

    return prealloc_matrix


def load_external_and_portfolios_from_excel(
    path: str,
    sheet_name: str,
    employees: List[Employee],
    col_portfolio: str = "Portefølje [t]",
    col_external: str = "Ekstern [t]",
) -> None:
    """
    Indlæser ekstern tid (undervisning) og portefølje pr. medarbejder
    fra arket 'Eksterne timer og porteføljer' og skriver værdierne ind
    i Employee-objekterne.

    Forventet struktur i arket:
      - Kolonne 'Medarbejder'
      - Kolonne col_portfolio (default 'Portefølje [t]')
      - Kolonne col_external  (default 'Ekstern [t]')
    """

    df = pd.read_excel(path, sheet_name=sheet_name)
    if "Medarbejder" not in df.columns:
        raise ValueError("Arket med eksterne timer skal have en kolonne 'Medarbejder'.")

    df = df.set_index("Medarbejder")

    if col_portfolio not in df.columns:
        raise ValueError(f"Kolonnen '{col_portfolio}' blev ikke fundet i arket '{sheet_name}'.")
    if col_external not in df.columns:
        raise ValueError(f"Kolonnen '{col_external}' blev ikke fundet i arket '{sheet_name}'.")

    emp_names = [e.name for e in employees]
    missing_emps = set(emp_names) - set(df.index)
    if missing_emps:
        raise ValueError(
            f"Følgende medarbejdere mangler i arket '{sheet_name}': {missing_emps}"
        )

    # Skriv værdierne direkte ind i Employee-objekterne
    for emp in employees:
        row = df.loc[emp.name]
        emp.portfolio_hours = float(row[col_portfolio])
        emp.teaching_hours = float(row[col_external])


def main():
    semester = "F2026"
    excel_path = "time_satser_F2026.xlsx"

    # Medarbejdere – kun navne her; portefølje/undervisning sættes fra Excel
    employees = [
        Employee("Søren Erbs Poulsen (SOEB)", portfolio_hours=0.0, teaching_hours=0.0),
        Employee("Mathias Larsen (MATL)",     portfolio_hours=0.0, teaching_hours=0.0),
        Employee("Rune Kier Nielsen (RUNI)",  portfolio_hours=0.0, teaching_hours=0.0),
        Employee("Kristoffer Bested Nielsen (KRI)", portfolio_hours=0.0, teaching_hours=0.0),
        Employee("Søren Andersen (SSSA)",     portfolio_hours=0.0, teaching_hours=0.0),
        Employee("Karl Woldum Tordrup (KART)", portfolio_hours=0.0, teaching_hours=0.0),
        Employee("Marton Major (MMAJ)",       portfolio_hours=0.0, teaching_hours=0.0),
        # NN – meget stor portefølje, ingen undervisning (sættes også fra Excel-arket)
        Employee("NN (ufordelt)",            portfolio_hours=0.0, teaching_hours=0.0),
    ]

    # Projekter – budget overskrives fra Excel
    projects = [
        Project("The Change",      budget=0.0),
        Project("LEG-DHC",         budget=0.0),
        Project("LTDE-repBC",      budget=0.0),
        Project("COOLGEOHEAT II",  budget=0.0),
        Project("HEATCODE",        budget=0.0),
    ]

    # 0) Indlæs eksterne timer og porteføljer fra arket "Eksterne timer og porteføljer"
    load_external_and_portfolios_from_excel(
        path=excel_path,
        sheet_name="Eksterne timer og porteføljer",
        employees=employees,
        col_portfolio="Portefølje [t]",   # tilpas hvis dine kolonnenavne er anderledes
        col_external="Ekstern [t]",
    )

    # 1) Indlæs timesatser og budgetter
    rate_matrix, budgets_arr = load_rates_and_budgets_from_excel(
        path=excel_path,
        sheet_name="Timesatser_budget",
        employees=employees,
        projects=projects,
    )

    # Opdatér projektbudgetter fra Excel
    for proj, b in zip(projects, budgets_arr):
        proj.budget = float(b)

    # 2) Indlæs preallokerede timer fra arket 'Preallokering'
    prealloc_matrix = load_preallocated_hours_from_excel(
        path=excel_path,
        sheet_name="Preallokering",
        employees=employees,
        projects=projects,
    )

    # Prioriteter – eksempelvis høj prioritet på bestemte kombinationer
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
            preallocated_hours=prealloc_matrix,
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
        source_excel_path=excel_path,        # inputfil med timesatser + budgetter
        source_sheet_name="Timesatser_budget",
    )
    print(f"Porteføljen er gemt i 'Portefølje_{semester}.xlsx'.")


if __name__ == "__main__":
    main()
