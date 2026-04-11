"""
Shared helper utilities for the diabetes readmission project.
"""

def print_step(step_name, df):
    """Print the current shape of the dataframe after a processing step."""
    print(f"[{step_name}] Shape: {df.shape[0]} rows x {df.shape[1]} columns")


def group_icd9(code):
    """
    Map an ICD-9 diagnosis code to a clinical category.

    ICD-9 codes follow standardized ranges for organ systems.
    We collapse hundreds of unique codes into ~10 meaningful groups
    to reduce cardinality while preserving clinical signal.
    """
    if code is None or str(code).strip() == '' or str(code) == 'nan':
        return 'Missing'

    code_str = str(code).strip()

    # Handle E and V codes (injury/supplementary)
    if code_str.startswith('E'):
        return 'Injury'
    if code_str.startswith('V'):
        return 'Supplementary'

    try:
        numeric = float(code_str)
    except ValueError:
        return 'Other'

    if 390 <= numeric <= 459 or numeric == 785:
        return 'Circulatory'
    elif 460 <= numeric <= 519 or numeric == 786:
        return 'Respiratory'
    elif 520 <= numeric <= 579 or numeric == 787:
        return 'Digestive'
    elif 250 <= numeric < 251:
        return 'Diabetes'
    elif 800 <= numeric <= 999:
        return 'Injury'
    elif 710 <= numeric <= 739:
        return 'Musculoskeletal'
    elif 580 <= numeric <= 629 or numeric == 788:
        return 'Genitourinary'
    elif 140 <= numeric <= 239:
        return 'Neoplasms'
    elif 240 <= numeric < 250 or 251 <= numeric <= 279:
        return 'Endocrine_Other'
    elif 680 <= numeric <= 709:
        return 'Skin'
    elif 1 <= numeric <= 139:
        return 'Infectious'
    elif 290 <= numeric <= 319:
        return 'Mental'
    elif 280 <= numeric <= 289:
        return 'Blood'
    elif 320 <= numeric <= 389:
        return 'Nervous'
    else:
        return 'Other'


# Age bracket midpoints for ordinal encoding
AGE_MAP = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95
}

# Medication dosage change encoding
MED_MAP = {'No': 0, 'Steady': 1, 'Down': 2, 'Up': 3}

# Lab result encodings
A1C_MAP = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}
GLU_MAP = {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}

# All 23 medication columns in the dataset
MEDICATION_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone'
]

# Discharge dispositions indicating patient died or went to hospice
DECEASED_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]
