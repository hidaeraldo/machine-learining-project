"""
Data loading and initial cleanup functions.
"""
import pandas as pd


def load_data(path, na_marker='?'):
    """
    Load the diabetic_data.csv file, replacing the custom missing marker
    with proper NaN values.

    We use keep_default_na=False so that the string "None" in columns
    like A1Cresult and max_glu_serum is kept as a valid category
    (meaning "test not performed") rather than converted to NaN.
    """
    df = pd.read_csv(path, na_values=[na_marker], keep_default_na=False,
                     low_memory=False)
    return df


def load_ids_mapping(path):
    """
    Parse the IDs_mapping.csv which has multiple tables separated by blank lines.
    Returns a dict of {table_name: DataFrame}.
    """
    mappings = {}
    current_name = None
    current_rows = []
    current_header = None

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_name and current_rows:
                    mappings[current_name] = pd.DataFrame(
                        current_rows, columns=current_header
                    )
                current_name = None
                current_rows = []
                current_header = None
                continue

            parts = line.split(',', 1)
            if current_name is None:
                current_header = [p.strip() for p in parts]
                current_name = current_header[0]
                current_rows = []
            else:
                current_rows.append([p.strip() for p in parts])

    if current_name and current_rows:
        mappings[current_name] = pd.DataFrame(
            current_rows, columns=current_header
        )

    return mappings
