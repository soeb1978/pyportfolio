from portfolio_model import (
    Employee, Project,
    allocate_hours_with_nn,
    print_allocation_report,
    export_portfolio_to_excel,
    AllocationError,
)

def main():
    semester = "F2026"  # Forår 2026

    employees = [
        Employee("Søren Erbs Poulsen (SOEB)", hourly_rate=656.64, portfolio_hours=657.5, teaching_hours=0),
        Employee("Mathias Larsen (MATL)",   hourly_rate=531.36, portfolio_hours=657.5, teaching_hours=450.0),   #150 på Ronja, Torben 300
        Employee("Rune Kier Nielsen (RUNI)", hourly_rate=560.14, portfolio_hours=1315/2-480, teaching_hours=0),
        Employee("Kristoffer Bested Nielsen (KRI)",  hourly_rate=499.35, portfolio_hours=657.5, teaching_hours=547.0),   
        Employee("Søren Andersen (SSSA)", hourly_rate=544.32, portfolio_hours=657.5, teaching_hours=500.0),           
        Employee("Karl Woldum Tordrup (KART)", hourly_rate=600.31, portfolio_hours=100, teaching_hours=0),
        Employee("Marton Major (MMAJ)", hourly_rate=529.01, portfolio_hours=657.5, teaching_hours=0),
    ]

    projects = [
        Project("The Change", budget=369_670),
        Project("LEG-DHC", budget=230_940),
        Project("LTDE-repBC", budget=398_530),
        Project("COOLGEOHEAT II", budget=299_881),
        Project("HEATCODE", budget=149_954),
    ]

    priorities = {
        ("Søren Erbs Poulsen (SOEB)", "LTDE-repBC"): 4.0,
        ("Søren Erbs Poulsen (SOEB)", "COOLGEOHEAT II"): 3.0,
        ("Søren Erbs Poulsen (SOEB)", "The Change"): 2.0,
        ("Kristoffer Bested Nielsen (KRI)", "COOLGEOHEAT II"): 3.0,
        ("Marton Major (MMAJ)", "LEG-DHC"): 3.0,
        ("Karl Woldum Tordrup (KART)", "The Change"): 3.0,
        ("Rune Kier Nielsen (RUNI)", "LTDE-repBC"): 2.0,
        ("Rune Kier Nielsen (RUNI)", "COOLGEOHEAT II"): 3.0,
        ("Søren Andersen (SSSA)", "HEATCODE"): 3.0,
        ("Mathias Larsen (MATL)", "The Change"): 3.0,
    }

    try:
        result = allocate_hours_with_nn(
            employees=employees,
            projects=projects,
            priorities=priorities,
            nn_name="NN",
            nn_hourly_rate=530.0,
            nn_max_hours=5_000.0,
        )
    except AllocationError as e:
        print("Porteføljeberegning fejlede:")
        print(e)
        return

    print_allocation_report(employees, projects, result)

    # Gem til Excel til administrationsbrug
    export_portfolio_to_excel(
        employees=employees,
        projects=projects,
        allocation_result=result,
        semester_label=semester,
        # filename kan udelades, så bliver det f.eks. "portefolje_F2026.xlsx"
        filename=f"Portefølje_{semester}.xlsx",
    )
    print("Porteføljen er gemt i "+f"Portefølje_{semester}.xlsx.")

if __name__ == "__main__":
    main()