import pandas as pd
import os
import io

files = [
    "EPA PFAS Master List V2.xlsx",
    "tx0c00264_si_003.xlsx"
]

report_file = "inspection_report.txt"

with open(report_file, 'w', encoding='utf-8') as f_out:
    f_out.write(f"Inspection Report\n=================\n\n")
    
    for filename in files:
        f_out.write(f"Processing: {filename}\n")
        f_out.write(f"{'-'*30}\n")
        
        if not os.path.exists(filename):
            f_out.write(f"ERROR: {filename} does not exist.\n\n")
            continue
            
        try:
            xl = pd.ExcelFile(filename)
            f_out.write(f"Sheet names: {xl.sheet_names}\n")
            
            for sheet_name in xl.sheet_names:
                f_out.write(f"\n  Sheet: '{sheet_name}'\n")
                df = pd.read_excel(filename, sheet_name=sheet_name, nrows=5)
                f_out.write(f"  Columns: {list(df.columns)}\n")
                
                # capture buffer
                buf = io.StringIO()
                df.to_string(buf=buf, index=False)
                f_out.write(f"  Data Head:\n{buf.getvalue()}\n")
                f_out.write("\n")
                
        except Exception as e:
            f_out.write(f"ERROR reading {filename}: {str(e)}\n")
        f_out.write("\n")

print(f"Report written to {report_file}")
