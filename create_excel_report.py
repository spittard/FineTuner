#!/usr/bin/env python3
"""
Create a formatted Excel report from control set results
"""

import re
import os

# Check for openpyxl, install if needed
try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Fill, PatternFill, Border, Side, Alignment
    from openpyxl.utils import get_column_letter
except ImportError:
    print("Installing openpyxl...")
    os.system('pip install openpyxl')
    from openpyxl import Workbook
    from openpyxl.styles import Font, Fill, PatternFill, Border, Side, Alignment
    from openpyxl.utils import get_column_letter


def parse_markdown(md_file):
    """Parse the markdown results file"""
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse case info
    case_pattern = r'### (\d+)\. (.+?)\n\n\*\*Top Match\*\*: (.+?)\n\n- \*\*Score\*\*: ([\d.]+) \(([\d.]+)%\)\n- \*\*Match Type\*\*: (\w+)\n- \*\*String Score\*\*: ([\d.]+)\n- \*\*Semantic Score\*\*: ([\d.]+)'
    cases = re.findall(case_pattern, content)
    
    # Parse tables (now with 6 columns: Rank, Name, Score, String, Semantic, Type)
    table_pattern = r'### (\d+)\. .+?\n.*?\|.*?\n\|[-\|]+\n((?:\|[^\n]+\n)+)'
    tables = re.findall(table_pattern, content, re.DOTALL)
    
    case_tables = {}
    for case_num, table_content in tables:
        rows = []
        for line in table_content.strip().split('\n'):
            if line.startswith('|'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 6:  # Now expecting 6 columns
                    rows.append(parts)
                elif len(parts) >= 4:  # Fallback for old format
                    # Add empty string/semantic scores
                    rows.append(parts[:3] + ['', ''] + parts[3:])
        case_tables[case_num] = rows
    
    return cases, case_tables


def create_process_diagram_sheet(wb):
    """Create a sheet with the matching process diagram"""
    ws = wb.create_sheet("Matching Process")
    
    # Styles
    title_font = Font(name='Segoe UI', size=16, bold=True, color='1A1A2E')
    header_font = Font(name='Segoe UI', size=12, bold=True, color='FFFFFF')
    section_font = Font(name='Segoe UI', size=11, bold=True, color='2E86AB')
    body_font = Font(name='Consolas', size=10)
    note_font = Font(name='Segoe UI', size=9, italic=True, color='6C757D')
    
    phase_fill = PatternFill(start_color='2E86AB', end_color='2E86AB', fill_type='solid')
    step_fill = PatternFill(start_color='E8F4F8', end_color='E8F4F8', fill_type='solid')
    formula_fill = PatternFill(start_color='FFF3CD', end_color='FFF3CD', fill_type='solid')
    example_fill = PatternFill(start_color='D4EDDA', end_color='D4EDDA', fill_type='solid')
    
    thin_border = Border(
        left=Side(style='thin', color='DEE2E6'),
        right=Side(style='thin', color='DEE2E6'),
        top=Side(style='thin', color='DEE2E6'),
        bottom=Side(style='thin', color='DEE2E6')
    )
    
    # Column widths
    ws.column_dimensions['A'].width = 3
    ws.column_dimensions['B'].width = 25
    ws.column_dimensions['C'].width = 50
    ws.column_dimensions['D'].width = 20
    ws.column_dimensions['E'].width = 25
    
    row = 1
    
    # Title
    ws.cell(row=row, column=2, value="Company Matcher - Scoring Algorithm").font = title_font
    ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=5)
    row += 2
    
    # Overview
    ws.cell(row=row, column=2, value="OVERVIEW").font = section_font
    row += 1
    ws.cell(row=row, column=2, value="Hybrid semantic + lexical matching with penalty-based re-ranking")
    row += 1
    ws.cell(row=row, column=2, value="Final Score = (String Score x 0.7) + (Semantic Score x 0.3)").font = Font(name='Consolas', size=10, bold=True)
    ws.cell(row=row, column=2).fill = formula_fill
    row += 2
    
    # Phase 1
    ws.cell(row=row, column=2, value="PHASE 1: SEMANTIC RETRIEVAL").font = header_font
    ws.cell(row=row, column=2).fill = phase_fill
    ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=5)
    row += 1
    
    phase1_steps = [
        ("Input", "Query string (e.g., 'National Air & Space Museum')"),
        ("Step", "Encode query using SentenceTransformer model"),
        ("Model", "paraphrase-MiniLM-L3-v2 (384 dimensions)"),
        ("Step", "Search FAISS index for top 50 candidates"),
        ("Output", "Raw Semantic Scores (dot product, range 0-10+)"),
    ]
    for label, desc in phase1_steps:
        ws.cell(row=row, column=2, value=label).font = Font(bold=True)
        ws.cell(row=row, column=2).fill = step_fill
        ws.cell(row=row, column=3, value=desc).font = body_font
        row += 1
    
    row += 1
    
    # Phase 2
    ws.cell(row=row, column=2, value="PHASE 2: STRING SIMILARITY SCORING").font = header_font
    ws.cell(row=row, column=2).fill = phase_fill
    ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=5)
    row += 1
    
    # Step 1
    ws.cell(row=row, column=2, value="Step 1: Weighted Jaccard").font = section_font
    row += 1
    step1_details = [
        ("Purpose", "Down-weight generic terms, up-weight distinctive words"),
        ("Generic Terms", "center(0.3), school(0.3), meeting(0.3), national(0.4)"),
        ("Distinctive", "Any word not in generic list gets weight 1.0"),
        ("Formula", "sum(intersection weights) / sum(union weights)"),
        ("Output", "base_score (0.0 - 1.0)"),
    ]
    for label, desc in step1_details:
        ws.cell(row=row, column=2, value=label).font = Font(bold=True, size=9)
        ws.cell(row=row, column=2).fill = step_fill
        ws.cell(row=row, column=3, value=desc).font = body_font
        if label == "Formula":
            ws.cell(row=row, column=3).fill = formula_fill
        row += 1
    
    row += 1
    
    # Step 2
    ws.cell(row=row, column=2, value="Step 2: Discriminating Word Penalty").font = section_font
    row += 1
    step2_details = [
        ("Purpose", "Penalize when query's unique words are missing from target"),
        ("Identifies", "Words with weight >= 0.8 (distinctive identifiers)"),
        ("Formula", "penalty = 1.0 - (missing_ratio x 0.4)"),
        ("Range", "0.6 - 1.0 (up to 40% reduction)"),
        ("Output", "base_score = base_score x penalty"),
    ]
    for label, desc in step2_details:
        ws.cell(row=row, column=2, value=label).font = Font(bold=True, size=9)
        ws.cell(row=row, column=2).fill = step_fill
        ws.cell(row=row, column=3, value=desc).font = body_font
        if label == "Formula":
            ws.cell(row=row, column=3).fill = formula_fill
        row += 1
    
    row += 1
    
    # Step 3
    ws.cell(row=row, column=2, value="Step 3: Short String Cap").font = section_font
    row += 1
    step3_details = [
        ("Purpose", "Prevent short matches from getting artificially high scores"),
        ("< 5 chars", "Cap at 50%"),
        ("5-7 chars", "Cap at 65%"),
        (">= 8 chars", "No cap"),
        ("Output", "base_score = min(base_score, cap)"),
    ]
    for label, desc in step3_details:
        ws.cell(row=row, column=2, value=label).font = Font(bold=True, size=9)
        ws.cell(row=row, column=2).fill = step_fill
        ws.cell(row=row, column=3, value=desc).font = body_font
        row += 1
    
    row += 1
    
    # Step 4
    ws.cell(row=row, column=2, value="Step 4: Category Mismatch Penalty").font = section_font
    row += 1
    step4_details = [
        ("Purpose", "Penalize when entity types differ"),
        ("Categories", "service_type: senior, medical, financial, legal, nursing"),
        ("Detection", "If both have category words but they differ"),
        ("Penalty", "0.75 (25% reduction per mismatch)"),
        ("Output", "base_score = base_score x category_penalty"),
    ]
    for label, desc in step4_details:
        ws.cell(row=row, column=2, value=label).font = Font(bold=True, size=9)
        ws.cell(row=row, column=2).fill = step_fill
        ws.cell(row=row, column=3, value=desc).font = body_font
        row += 1
    
    row += 1
    
    # Step 5
    ws.cell(row=row, column=2, value="Step 5: Proper Noun Penalty").font = section_font
    row += 1
    step5_details = [
        ("Purpose", "Penalize when identifying names differ"),
        ("Detection", "Capitalized words not in COMMON_WORDS list"),
        ("Example", "'Hartford' vs 'Jefferson' = different proper nouns"),
        ("Penalty", "1.0 - (missing_ratio x 0.25)"),
        ("Output", "base_score = base_score x proper_noun_penalty"),
    ]
    for label, desc in step5_details:
        ws.cell(row=row, column=2, value=label).font = Font(bold=True, size=9)
        ws.cell(row=row, column=2).fill = step_fill
        ws.cell(row=row, column=3, value=desc).font = body_font
        if label == "Penalty":
            ws.cell(row=row, column=3).fill = formula_fill
        row += 1
    
    row += 2
    
    # Phase 3
    ws.cell(row=row, column=2, value="PHASE 3: FINAL SCORE CALCULATION").font = header_font
    ws.cell(row=row, column=2).fill = phase_fill
    ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=5)
    row += 1
    
    phase3_details = [
        ("Normalize", "Semantic Score: raw_score / max_score_in_batch"),
        ("Combine", "Final = (String x 0.7) + (Semantic x 0.3)"),
        ("Override", "Exact matches forced to 1.0"),
        ("Output", "Sorted list of matches with scores"),
    ]
    for label, desc in phase3_details:
        ws.cell(row=row, column=2, value=label).font = Font(bold=True)
        ws.cell(row=row, column=2).fill = step_fill
        ws.cell(row=row, column=3, value=desc).font = body_font
        if label == "Combine":
            ws.cell(row=row, column=3).fill = formula_fill
        row += 1
    
    row += 2
    
    # Example
    ws.cell(row=row, column=2, value="EXAMPLE: Score Breakdown").font = header_font
    ws.cell(row=row, column=2).fill = PatternFill(start_color='28A745', end_color='28A745', fill_type='solid')
    ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=5)
    row += 1
    
    ws.cell(row=row, column=2, value="Query:").font = Font(bold=True)
    ws.cell(row=row, column=3, value="North Shore Senior Center")
    row += 1
    ws.cell(row=row, column=2, value="Candidate:").font = Font(bold=True)
    ws.cell(row=row, column=3, value="North Shore Medical Center")
    row += 2
    
    example_calc = [
        ("Step 1", "Weighted Jaccard", "{north(0.5)+shore(0.5)+center(0.3)} / {all} = 1.3/3.3", "0.394"),
        ("Step 2", "Discrim. Penalty", "No missing distinctive words", "1.0"),
        ("Step 3", "Short String Cap", ">8 chars matched", "No cap"),
        ("Step 4", "Category Penalty", "senior != medical (service_type mismatch)", "0.75"),
        ("Step 5", "Proper Noun", "No proper nouns detected", "1.0"),
        ("", "String Score", "0.394 x 1.0 x 0.75 x 1.0", "0.296"),
        ("", "Semantic (norm)", "~0.95", "0.95"),
        ("", "FINAL", "(0.296 x 0.7) + (0.95 x 0.3)", "~49%"),
    ]
    
    headers = ["Step", "Component", "Calculation", "Value"]
    for col, h in enumerate(headers, 2):
        ws.cell(row=row, column=col, value=h).font = header_font
        ws.cell(row=row, column=col).fill = phase_fill
    row += 1
    
    for step, comp, calc, val in example_calc:
        ws.cell(row=row, column=2, value=step).font = body_font
        ws.cell(row=row, column=3, value=comp).font = body_font
        ws.cell(row=row, column=4, value=calc).font = body_font
        ws.cell(row=row, column=5, value=val).font = Font(name='Consolas', size=10, bold=True)
        if step == "":
            for c in range(2, 6):
                ws.cell(row=row, column=c).fill = example_fill
        row += 1
    
    row += 2
    
    # Thresholds
    ws.cell(row=row, column=2, value="RECOMMENDED THRESHOLDS").font = header_font
    ws.cell(row=row, column=2).fill = phase_fill
    ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=5)
    row += 1
    
    thresholds = [
        (">= 95%", "Auto-merge safe", "Exact matches, punctuation variants"),
        ("90-94%", "High confidence", "Word reordering, abbreviations"),
        ("80-89%", "Review recommended", "Corporate siblings, structural matches"),
        ("70-79%", "Manual review", "May include false positives"),
        ("< 70%", "Low confidence", "Likely unrelated entities"),
    ]
    
    for score, action, examples in thresholds:
        ws.cell(row=row, column=2, value=score).font = Font(bold=True)
        ws.cell(row=row, column=3, value=action).font = body_font
        ws.cell(row=row, column=4, value=examples).font = note_font
        row += 1
    
    # Freeze top row
    ws.freeze_panes = 'A3'


def create_excel_report(md_file, excel_file):
    """Create a beautifully formatted Excel report"""
    
    cases, case_tables = parse_markdown(md_file)
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Control Set Results"
    
    # Define styles
    header_font = Font(name='Segoe UI', size=12, bold=True, color='FFFFFF')
    header_fill = PatternFill(start_color='2E86AB', end_color='2E86AB', fill_type='solid')
    
    query_font = Font(name='Segoe UI', size=11, bold=True, color='1A1A2E')
    query_fill = PatternFill(start_color='E8F4F8', end_color='E8F4F8', fill_type='solid')
    
    match_font = Font(name='Consolas', size=10)
    exact_fill = PatternFill(start_color='D4EDDA', end_color='D4EDDA', fill_type='solid')
    hybrid_fill = PatternFill(start_color='FFFFFF', end_color='FFFFFF', fill_type='solid')
    alt_fill = PatternFill(start_color='F8F9FA', end_color='F8F9FA', fill_type='solid')
    
    score_high_font = Font(name='Segoe UI', size=10, bold=True, color='155724')
    score_med_font = Font(name='Segoe UI', size=10, color='856404')
    score_low_font = Font(name='Segoe UI', size=10, color='721C24')
    
    thin_border = Border(
        left=Side(style='thin', color='DEE2E6'),
        right=Side(style='thin', color='DEE2E6'),
        top=Side(style='thin', color='DEE2E6'),
        bottom=Side(style='thin', color='DEE2E6')
    )
    
    thick_border = Border(
        left=Side(style='medium', color='2E86AB'),
        right=Side(style='medium', color='2E86AB'),
        top=Side(style='medium', color='2E86AB'),
        bottom=Side(style='medium', color='2E86AB')
    )
    
    # Set column widths
    ws.column_dimensions['A'].width = 8
    ws.column_dimensions['B'].width = 40
    ws.column_dimensions['C'].width = 40
    ws.column_dimensions['D'].width = 10
    ws.column_dimensions['E'].width = 10
    ws.column_dimensions['F'].width = 10
    ws.column_dimensions['G'].width = 9
    
    # Create header row
    headers = ['Case', 'Query', 'Match', 'Score', 'String', 'Semantic', 'Type']
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = thin_border
    
    ws.row_dimensions[1].height = 25
    
    # Freeze header row
    ws.freeze_panes = 'A2'
    
    # Add data
    current_row = 2
    
    for case in cases:
        case_num, query, top_match, score, score_pct, match_type, string_score, semantic_score = case
        
        table_rows = case_tables.get(case_num, [])
        num_matches = len(table_rows)
        
        # Query cell (merged for all matches)
        query_cell = ws.cell(row=current_row, column=1, value=int(case_num))
        query_cell.font = query_font
        query_cell.fill = query_fill
        query_cell.alignment = Alignment(horizontal='center', vertical='top')
        query_cell.border = thin_border
        
        query_name_cell = ws.cell(row=current_row, column=2, value=query)
        query_name_cell.font = query_font
        query_name_cell.fill = query_fill
        query_name_cell.alignment = Alignment(horizontal='left', vertical='top', wrap_text=True)
        query_name_cell.border = thin_border
        
        # Merge query cells if multiple matches
        if num_matches > 1:
            ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row + num_matches - 1, end_column=1)
            ws.merge_cells(start_row=current_row, start_column=2, end_row=current_row + num_matches - 1, end_column=2)
        
        # Add matches
        for i, match_row in enumerate(table_rows):
            # New format: rank, name, score, string_score, semantic_score, type
            rank, name, match_score, string_score, semantic_score, mtype = match_row
            row = current_row + i
            
            # Match name
            match_cell = ws.cell(row=row, column=3, value=name)
            match_cell.font = match_font
            match_cell.alignment = Alignment(horizontal='left', vertical='center')
            match_cell.border = thin_border
            
            # Score
            score_val = float(match_score.replace('%', ''))
            score_cell = ws.cell(row=row, column=4, value=match_score)
            score_cell.alignment = Alignment(horizontal='center', vertical='center')
            score_cell.border = thin_border
            
            if score_val >= 90:
                score_cell.font = score_high_font
            elif score_val >= 70:
                score_cell.font = score_med_font
            else:
                score_cell.font = score_low_font
            
            # String Score
            string_cell = ws.cell(row=row, column=5, value=string_score)
            string_cell.alignment = Alignment(horizontal='center', vertical='center')
            string_cell.border = thin_border
            string_cell.font = Font(name='Consolas', size=9, color='495057')
            
            # Semantic Score
            semantic_cell = ws.cell(row=row, column=6, value=semantic_score)
            semantic_cell.alignment = Alignment(horizontal='center', vertical='center')
            semantic_cell.border = thin_border
            semantic_cell.font = Font(name='Consolas', size=9, color='6C757D')
            
            # Type
            type_cell = ws.cell(row=row, column=7, value=mtype)
            type_cell.alignment = Alignment(horizontal='center', vertical='center')
            type_cell.border = thin_border
            
            # Row fill based on match type and alternating
            all_cells = [match_cell, score_cell, string_cell, semantic_cell, type_cell]
            if mtype == 'exact':
                for c in all_cells:
                    c.fill = exact_fill
            elif i % 2 == 0:
                for c in all_cells:
                    c.fill = hybrid_fill
            else:
                for c in all_cells:
                    c.fill = alt_fill
        
        current_row += num_matches
        
        # Add separator row
        for col in range(1, 8):
            sep_cell = ws.cell(row=current_row, column=col, value='')
            sep_cell.fill = PatternFill(start_color='DEE2E6', end_color='DEE2E6', fill_type='solid')
        ws.row_dimensions[current_row].height = 3
        current_row += 1
    
    # Add summary sheet
    ws_summary = wb.create_sheet("Summary")
    
    # Summary headers
    summary_headers = ['Case', 'Query', 'Top Score', 'Type', 'String Score', 'Semantic Score']
    for col, header in enumerate(summary_headers, 1):
        cell = ws_summary.cell(row=1, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = thin_border
    
    ws_summary.column_dimensions['A'].width = 8
    ws_summary.column_dimensions['B'].width = 50
    ws_summary.column_dimensions['C'].width = 12
    ws_summary.column_dimensions['D'].width = 10
    ws_summary.column_dimensions['E'].width = 14
    ws_summary.column_dimensions['F'].width = 14
    
    ws_summary.freeze_panes = 'A2'
    
    for row_num, case in enumerate(cases, 2):
        case_num, query, top_match, score, score_pct, match_type, string_score, semantic_score = case
        
        ws_summary.cell(row=row_num, column=1, value=int(case_num)).alignment = Alignment(horizontal='center')
        ws_summary.cell(row=row_num, column=2, value=query)
        ws_summary.cell(row=row_num, column=3, value=score_pct + '%').alignment = Alignment(horizontal='center')
        ws_summary.cell(row=row_num, column=4, value=match_type).alignment = Alignment(horizontal='center')
        ws_summary.cell(row=row_num, column=5, value=float(string_score)).alignment = Alignment(horizontal='center')
        ws_summary.cell(row=row_num, column=6, value=float(semantic_score)).alignment = Alignment(horizontal='center')
        
        # Alternating row colors
        fill = alt_fill if row_num % 2 == 0 else hybrid_fill
        for col in range(1, 7):
            ws_summary.cell(row=row_num, column=col).fill = fill
            ws_summary.cell(row=row_num, column=col).border = thin_border
    
    # Add Process Diagram sheet
    create_process_diagram_sheet(wb)
    
    # Save
    wb.save(excel_file)
    print(f"[OK] Created {excel_file}")
    print(f"  - Sheet 1: 'Control Set Results' (all matches, formatted)")
    print(f"  - Sheet 2: 'Summary' (one row per query)")
    print(f"  - Sheet 3: 'Matching Process' (algorithm diagram)")


if __name__ == '__main__':
    create_excel_report(
        'companies_control_set_results.md',
        'Control_Set_Results.xlsx'
    )

