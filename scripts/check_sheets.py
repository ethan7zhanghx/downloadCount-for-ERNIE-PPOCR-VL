import pandas as pd
import sys

def get_sheet_names(file_path):
    """Prints the sheet names of an Excel file."""
    try:
        xls = pd.ExcelFile(file_path)
        print(f"Sheet names for {file_path}:")
        for sheet_name in xls.sheet_names:
            print(f"- {sheet_name}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        get_sheet_names(sys.argv[1])
    else:
        print("Usage: python check_sheets.py <path_to_excel_file>")
