import argparse

def get_parser() -> argparse.ArgumentParser:
    """
    Define the parse arguments from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Argument parsers."
    )
    # Parse the start_date and end_date arguments
    parser.add_argument(
        "-sd", 
        "--start_date",
        action="store",
        help="Start date of the data to load (e.g.: 2023-04-30 11:05:00).",
        type=str
    )
    parser.add_argument(
        "-ed",
        "--end_date",
        action="store",
        help="End date of the data to load (e.g.: 2023-04-30 11:05:00).",
        type=str
    )
    
    # Parse the latitude and longitude arguments
    parser.add_argument(
        "-lat",
        "--latitude",
        action="store",
        help="Latitude of the location to load the data.",
        type=float
    )
    parser.add_argument(
        "-lon",
        "--longitude",
        action="store",
        help="Longitude of the location to load the data.",
        type=float
    )
    return parser