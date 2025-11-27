from portfolio_model import (
    Employee, Project,
    allocate_hours_with_nn,
    print_allocation_report,
    AllocationError,
)

def main():
    employees = [
        Employee("Søen Erbs Poulsen (SOEB)", hourly_rate=656.64, portfolio_hours=657.5, teaching_hours=0),
        Employee("Mathias Larsen (MATL)",   hourly_rate=531.36, portfolio_hours=657.5, teaching_hours=450.0),   #150 på Ronja, Torben 300
    ]

    projects = [
        Project("Projekt A", budget=400_000),
        Project("Projekt B", budget=250_000),
    ]

    priorities = {
        ("Alice", "Projekt A"): 3.0,
        ("Bob",   "Projekt B"): 3.0,
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
