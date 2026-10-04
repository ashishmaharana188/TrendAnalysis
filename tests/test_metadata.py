from data_access.metadata import (
    get_active_companies,
    get_company,
    get_sectors,
    get_industries,
)


def main() -> None:
    companies = get_active_companies()

    print(f"Active companies: {len(companies)}")

    print("\nFirst 5 companies:")
    for company in companies[:5]:
        print(
            company["Ticker"],
            "|",
            company["Sector"],
            "|",
            company["Industry"],
        )

    print("\nCompany lookup:")
    print(get_company("RELIANCE"))

    sectors = get_sectors()
    print(f"\nSectors: {len(sectors)}")
    print(sectors)

    industries = get_industries()
    print(f"\nIndustries: {len(industries)}")
    print(industries)


if __name__ == "__main__":
    main()