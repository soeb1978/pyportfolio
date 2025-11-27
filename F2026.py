from portfolio_model import (
    Employee, Project,
    allocate_hours_with_nn,
    print_allocation_report,
    AllocationError,
)

def main():
    employees = [
        Employee("Søren Erbs Poulsen (SOEB)", hourly_rate=656.64, portfolio_hours=657.5, teaching_hours=0),
        Employee("Mathias Larsen (MATL)",   hourly_rate=531.36, portfolio_hours=657.5, teaching_hours=450.0),   #150 på Ronja, Torben 300
        Employee("Rune Kier Nielsen (RUNI)", hourly_rate=560.14, portfolio_hours=657.5, teaching_hours=0),
        Employee("Kristoffer Bested Nielsen (KRI)",  hourly_rate=499.35, portfolio_hours=657.5, teaching_hours=547.0),   
        Employee("Søren Andersen (SSSA)", hourly_rate=544.32, portfolio_hours=657.5, teaching_hours=500.0),           
        Employee("Karl Woldum Tordrup (KART)", hourly_rate=600.31, portfolio_hours=350, teaching_hours=0),
        Employee("Marton Major (MMAJ)", hourly_rate=529.01, portfolio_hours=657.5, teaching_hours=0),
    ]

    projects = [
        Project("The Change", budget=369_670),
        Project("LEG-DHC", budget=230_940),
        Project("LTDE-repBC", budget=398_530),
        Project("COOLGEOHEAT II", budget=250_000),
        Project("HEATCODE", budget=250_000),
    ]

    priorities = {
        ("Søren Erbs Poulsen (SOEB)", "COOLGEOHEAT II"): 3.0,
        ("Kristoffer Bested Nielsen (KRI)", "COOLGEOHEAT II"): 3.0,
    }

    try:
        result = allocate_hours_with_nn(
            employees=employees,
            projects=projects,
            priorities=priorities,
            nn_name="NN",
            nn_hourly_rate=600.0,
            nn_max_hours=5_000.0,
        )
    except AllocationError as e:
        print("Porteføljeberegning fejlede:")
        print(e)
        return

    print_allocation_report(employees, projects, result)


if __name__ == "__main__":
    main()
