import pandas as pd
import sys

# Files
pfas_file = "EPA PFAS Master List V2.xlsx"
tox21_file = "tx0c00264_si_003.xlsx"
output_file = "overlap_results.txt"

with open(output_file, 'w') as f_out:
    f_out.write("Overlap Analysis Report\n=======================\n")
    
    print("Loading EPA PFAS Master List...")
    f_out.write("\nDataset 1: EPA PFAS Master List\n")
    df_pfas = pd.read_excel(pfas_file, sheet_name='EPA PFAS Master List V2', usecols=['DTXSID', 'Chemical Name', 'CASRN'])
    pfas_ids = set(df_pfas['DTXSID'].dropna())
    f_out.write(f"- Total Rows: {len(df_pfas)}\n")
    f_out.write(f"- Unique DTXSIDs: {len(pfas_ids)}\n")

    print("Loading Tox21 Data (Sheet S2)...")
    f_out.write("\nDataset 2: Tox21 Data (Sheet S2)\n")
    df_tox21 = pd.read_excel(tox21_file, sheet_name='S2.TOX21S', usecols=['DTXSID', 'PREFERRED_NAME', 'Structure_SMILES'])
    tox21_ids = set(df_tox21['DTXSID'].dropna())
    f_out.write(f"- Total Rows: {len(df_tox21)}\n")
    f_out.write(f"- Unique DTXSIDs: {len(tox21_ids)}\n")

    # Calculate Overlap
    common_ids = pfas_ids.intersection(tox21_ids)
    f_out.write("\nIntersection Results:\n")
    f_out.write(f"- Common Chemicals (by DTXSID): {len(common_ids)}\n")
    percentage = (len(common_ids)/len(pfas_ids))*100 if len(pfas_ids) > 0 else 0
    f_out.write(f"- Coverage of PFAS List: {percentage:.2f}%\n")

    if len(common_ids) > 0:
        f_out.write("\nSample Common Chemicals:\n")
        sample = df_pfas[df_pfas['DTXSID'].isin(list(common_ids)[:5])][['DTXSID', 'Chemical Name']]
        f_out.write(sample.to_string(index=False))
        f_out.write("\n")

    # Check for predicted properties availability for these common chemicals
    print("Checking for predicted properties...")
    f_out.write("\nPredicted Properties Availability (Sheet S4):\n")
    df_props = pd.read_excel(tox21_file, sheet_name='S4.Predicted properties', usecols=['DTXSID', 'AMES_MUTAGENICITY_TEST_PRED', 'RatCarc_DEREK', 'ORAL_RAT_LD50_MOL/KG_TEST_PRED'])
    props_overlap = df_props[df_props['DTXSID'].isin(common_ids)]
    f_out.write(f"- Chemicals with Property predictions: {len(props_overlap)}\n")

print(f"Analysis complete. Results written to {output_file}")
