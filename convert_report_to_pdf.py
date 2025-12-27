import markdown
from xhtml2pdf import pisa
import os

# Configuration: Define the two requested reports
REPORTS = {
    "Project_Results_Detailed.pdf": ["all_result.md", "technology_stack.md"],
    "Comparison_Report.pdf": ["comparison_report.md"]
}

BASE_DIR = os.getcwd()

def convert_md_to_pdf(report_name, input_files):
    combined_html = ""
    
    # Minimal CSS to avoid library errors
    css_style = """
    <style>
        body { font-family: Helvetica, sans-serif; font-size: 10pt; }
        h1 { font-size: 18pt; color: #333; border-bottom: 2px solid #333; padding-bottom: 5px; margin-top: 30px; }
        h2 { font-size: 14pt; color: #555; margin-top: 20px; }
        img { max-width: 100%; height: auto; margin: 10px 0; }
        code { font-family: Courier; background: #eee; }
        pre { background: #eee; padding: 10px; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        td, th { border: 1px solid #ccc; padding: 5px; }
        .page-break { page-break-before: always; }
    </style>
    """

    full_html = f"<html><head>{css_style}</head><body>"
    
    # 1. Processing content
    for i, input_file in enumerate(input_files):
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping.")
            continue
            
        print(f"Adding {input_file} to {report_name}...")
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
            
        if i > 0:
            full_html += '<div class="page-break"></div>'
            
        # Convert Markdown to HTML
        # Using basic extensions only
        html_segment = markdown.markdown(text, extensions=['tables', 'fenced_code'])
        full_html += f'<div class="section">{html_segment}</div>'

    full_html += "</body></html>"

    # 2. Try Generating PDF
    print(f"Generating PDF: {report_name}...")
    try:
        with open(report_name, "wb") as result_file:
            pisa_status = pisa.CreatePDF(full_html, dest=result_file)
            
        if pisa_status.err:
            print(f"Error generating PDF {report_name}: {pisa_status.err}")
        else:
            print(f"SUCCESS: {report_name} created.")
            
    except Exception as e:
        print(f"CRITICAL ERROR generating PDF for {report_name}: {e}")
        # Build HTML fallback immediately
        html_fallback = report_name.replace(".pdf", ".html")
        with open(html_fallback, "w", encoding="utf-8") as f:
            f.write(full_html)
        print(f"Created HTML fallback: {html_fallback}")

if __name__ == "__main__":
    for report_name, inputs in REPORTS.items():
        print(f"--- Processing {report_name} ---")
        try:
            convert_md_to_pdf(report_name, inputs)
        except Exception as e:
            print(f"Failed to process {report_name}: {e}")
        print("\n")
