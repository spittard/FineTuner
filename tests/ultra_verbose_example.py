#!/usr/bin/env python3
"""
ULTRA-VERBOSE Verification Script - One example with copious explanations.

This script generates an extremely detailed report for a single query,
explaining every aspect of the scoring with actual data.
"""

import json
import os
import sys
import time
from datetime import datetime

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from finetuner.web.services.search_service import SearchService
from finetuner.web.services.rationale_service import RationaleService


def generate_ultra_verbose_report(query, matches, output_file='ultra_verbose_example.md'):
    """Generate an extremely detailed, verbose report for a single query"""
    
    md = []
    md.append(f"# Ultra-Verbose Match Analysis: \"{query}\"\n\n")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    md.append("> [!IMPORTANT]\n")
    md.append("> This report provides COPIOUS verbal explanations with actual data to thoroughly explain\n")
    md.append("> why each match is positioned where it is. Every score component is explained in detail.\n\n")
    
    md.append("---\n\n")
    
    # Overview section
    md.append("## Query Overview\n\n")
    md.append(f"**Search Query:** `{query}`\n\n")
    md.append(f"**Total Matches Found:** {len(matches)}\n\n")
    md.append(f"**Top Match:** {matches[0]['company_name']} ({matches[0]['likeness_percent']:.1f}%)\n\n")
    
    # Analyze each of top 10 matches in extreme detail
    num_to_analyze = min(10, len(matches))
    
    md.append(f"## Detailed Analysis of Top {num_to_analyze} Matches\n\n")
    md.append("Each match below includes:\n")
    md.append("1. Complete score breakdown with actual values\n")
    md.append("2. Detailed explanation of each scoring component\n")
    md.append("3. Verbose explanation of why it ranks where it does\n")
    md.append("4. Comparison with adjacent matches showing exact differences\n\n")
    md.append("---\n\n")
    
    for i, match in enumerate(matches[:num_to_analyze], 1):
        md.append(f"### Match #{i}: {match['company_name']}\n\n")
        
        # Extract all score components
        final_score = match.get('raw_score', match.get('score', 0.0))
        string_score = match.get('string_score', 0.0)
        semantic_raw = match.get('semantic_score', 0.0)
        semantic_norm = match.get('normalized_semantic_score', semantic_raw)
        acronym_fidelity = match.get('acronym_fidelity', 0.0)
        location_score = match.get('location_score', 0.0)
        match_type = match.get('match_type', 'hybrid')
        
        # Score summary box
        md.append("#### Score Summary\n\n")
        md.append("```\n")
        md.append(f"FINAL SCORE:     {final_score:.6f}  ({final_score*100:.2f}%)\n")
        md.append(f"RANK:            #{i} out of {len(matches)}\n")
        md.append(f"MATCH TYPE:      {match_type}\n")
        md.append("```\n\n")
        
        # Complete score breakdown table
        md.append("#### Complete Score Breakdown\n\n")
        md.append("| Component | Raw Value | Weight | Weighted Contribution | Percentage |\n")
        md.append("|-----------|-----------|--------|----------------------|------------|\n")
        
        string_contrib = string_score * 0.7
        md.append(f"| String Similarity | {string_score:.6f} | 70% | {string_contrib:.6f} | {string_contrib*100:.2f}% |\n")
        
        sem_contrib = semantic_norm * 0.3
        md.append(f"| Semantic Similarity (Norm) | {semantic_norm:.6f} | 30% | {sem_contrib:.6f} | {sem_contrib*100:.2f}% |\n")
        md.append(f"| Semantic Similarity (Raw) | {semantic_raw:.6f} | - | - | - |\n")
        
        base_score = string_contrib + sem_contrib
        md.append(f"| **Base Score** | **{base_score:.6f}** | - | - | **{base_score*100:.2f}%** |\n")
        
        if acronym_fidelity > 0.0:
            acro_contrib = acronym_fidelity * 0.15
            md.append(f"| Acronym Fidelity Boost | {acronym_fidelity:.6f} | 15% max | +{acro_contrib:.6f} | +{acro_contrib*100:.2f}% |\n")
        
        if location_score > 0.0:
            loc_contrib = location_score * 0.05
            md.append(f"| Location Match Boost | {location_score:.6f} | 5% max | +{loc_contrib:.6f} | +{loc_contrib*100:.2f}% |\n")
        
        md.append(f"| **FINAL SCORE** | **{final_score:.6f}** | - | - | **{final_score*100:.2f}%** |\n\n")
        
        # Verbose explanation of score calculation
        md.append("#### Detailed Score Calculation Explanation\n\n")
        
        md.append(f"**Step 1: String Similarity Component**\n\n")
        md.append(f"The string similarity score measures how closely the query text `\"{query}\"` matches ")
        md.append(f"the company name `\"{match['company_name']}\"` using lexical (word-based) comparison.\n\n")
        md.append(f"- **Raw String Score:** {string_score:.6f}\n")
        md.append(f"- **Weight Applied:** 70% (this is the primary scoring component)\n")
        md.append(f"- **Weighted Contribution:** {string_score:.6f} × 0.70 = {string_contrib:.6f}\n")
        md.append(f"- **Percentage Contribution:** {string_contrib*100:.2f}% of the final score\n\n")
        
        if string_score >= 0.95:
            md.append(f"**Interpretation:** EXCELLENT - The string similarity of {string_score:.6f} indicates nearly perfect lexical matching. ")
            md.append(f"The words in the query align very closely with the company name, suggesting this is likely the exact company or a very close variant.\n\n")
        elif string_score >= 0.80:
            md.append(f"**Interpretation:** VERY GOOD - The string similarity of {string_score:.6f} indicates strong lexical matching. ")
            md.append(f"Most words align well between the query and company name, though there may be some minor differences in word order or additional words.\n\n")
        elif string_score >= 0.60:
            md.append(f"**Interpretation:** GOOD - The string similarity of {string_score:.6f} indicates moderate lexical matching. ")
            md.append(f"There is significant word overlap, but also notable differences that prevent a higher score.\n\n")
        elif string_score >= 0.40:
            md.append(f"**Interpretation:** FAIR - The string similarity of {string_score:.6f} indicates partial lexical matching. ")
            md.append(f"Some words match, but there are substantial differences between the query and company name.\n\n")
        else:
            md.append(f"**Interpretation:** WEAK - The string similarity of {string_score:.6f} indicates minimal lexical matching. ")
            md.append(f"Very few words match directly, suggesting this match relies more on semantic understanding.\n\n")
        
        md.append(f"**Step 2: Semantic Similarity Component**\n\n")
        md.append(f"The semantic similarity score measures the meaning-based relationship between the query and company name ")
        md.append(f"using AI embeddings. This captures conceptual similarity even when exact words don't match.\n\n")
        md.append(f"- **Raw Semantic Score:** {semantic_raw:.6f} (from embedding model)\n")
        md.append(f"- **Normalized Semantic Score:** {semantic_norm:.6f} (scaled for consistency)\n")
        md.append(f"- **Weight Applied:** 30% (secondary to string matching)\n")
        md.append(f"- **Weighted Contribution:** {semantic_norm:.6f} × 0.30 = {sem_contrib:.6f}\n")
        md.append(f"- **Percentage Contribution:** {sem_contrib*100:.2f}% of the final score\n\n")
        
        if semantic_norm >= 0.90:
            md.append(f"**Interpretation:** EXCELLENT - The semantic similarity of {semantic_norm:.6f} indicates a very strong meaning-based connection. ")
            md.append(f"The AI model recognizes these as highly related concepts, even if the exact words differ.\n\n")
        elif semantic_norm >= 0.70:
            md.append(f"**Interpretation:** VERY GOOD - The semantic similarity of {semantic_norm:.6f} indicates a strong meaning-based connection. ")
            md.append(f"The concepts are closely related in the embedding space.\n\n")
        elif semantic_norm >= 0.50:
            md.append(f"**Interpretation:** GOOD - The semantic similarity of {semantic_norm:.6f} indicates a moderate meaning-based connection. ")
            md.append(f"There is conceptual overlap, though the relationship is not as strong as higher scores.\n\n")
        elif semantic_norm >= 0.30:
            md.append(f"**Interpretation:** FAIR - The semantic similarity of {semantic_norm:.6f} indicates some meaning-based connection. ")
            md.append(f"The AI model detects a relationship, but it's relatively weak.\n\n")
        else:
            md.append(f"**Interpretation:** WEAK - The semantic similarity of {semantic_norm:.6f} indicates minimal meaning-based connection. ")
            md.append(f"The concepts are not closely related in the embedding space.\n\n")
        
        md.append(f"**Step 3: Base Score Calculation**\n\n")
        md.append(f"The base score combines string and semantic components:\n\n")
        md.append(f"```\n")
        md.append(f"Base Score = (String × 0.70) + (Semantic × 0.30)\n")
        md.append(f"Base Score = ({string_score:.6f} × 0.70) + ({semantic_norm:.6f} × 0.30)\n")
        md.append(f"Base Score = {string_contrib:.6f} + {sem_contrib:.6f}\n")
        md.append(f"Base Score = {base_score:.6f}\n")
        md.append(f"```\n\n")
        
        md.append(f"This base score of {base_score:.6f} ({base_score*100:.2f}%) represents the core matching strength ")
        md.append(f"before any bonus adjustments are applied.\n\n")
        
        if acronym_fidelity > 0.0:
            acro_contrib = acronym_fidelity * 0.15
            md.append(f"**Step 4: Acronym Fidelity Boost**\n\n")
            md.append(f"An acronym relationship was detected between the query and this company name.\n\n")
            md.append(f"- **Acronym Fidelity Score:** {acronym_fidelity:.6f}\n")
            md.append(f"- **Maximum Boost:** 15% (0.15)\n")
            md.append(f"- **Actual Boost Applied:** {acronym_fidelity:.6f} × 0.15 = {acro_contrib:.6f}\n")
            md.append(f"- **Percentage Boost:** +{acro_contrib*100:.2f}%\n\n")
            
            if acronym_fidelity >= 0.90:
                md.append(f"**Interpretation:** EXCELLENT - The acronym fidelity of {acronym_fidelity:.6f} indicates this is highly likely ")
                md.append(f"to be a literal acronym expansion. The letters match precisely and in order.\n\n")
            elif acronym_fidelity >= 0.70:
                md.append(f"**Interpretation:** GOOD - The acronym fidelity of {acronym_fidelity:.6f} indicates this is probably ")
                md.append(f"an acronym expansion, though there may be some minor discrepancies.\n\n")
            else:
                md.append(f"**Interpretation:** MODERATE - The acronym fidelity of {acronym_fidelity:.6f} indicates a possible ")
                md.append(f"acronym connection, but it's not a perfect match.\n\n")
        
        if location_score > 0.0:
            loc_contrib = location_score * 0.05
            md.append(f"**Step {4 if acronym_fidelity == 0 else 5}: Location Match Boost**\n\n")
            md.append(f"Geographic information was available and matched between query and company.\n\n")
            md.append(f"- **Location Score:** {location_score:.6f}\n")
            md.append(f"- **Maximum Boost:** 5% (0.05)\n")
            md.append(f"- **Actual Boost Applied:** {location_score:.6f} × 0.05 = {loc_contrib:.6f}\n")
            md.append(f"- **Percentage Boost:** +{loc_contrib*100:.2f}%\n\n")
        
        final_step = 4
        if acronym_fidelity > 0.0:
            final_step += 1
        if location_score > 0.0:
            final_step += 1
            
        md.append(f"**Step {final_step}: Final Score**\n\n")
        md.append(f"```\n")
        md.append(f"Final Score = Base Score")
        if acronym_fidelity > 0.0:
            md.append(f" + Acronym Boost")
        if location_score > 0.0:
            md.append(f" + Location Boost")
        md.append(f"\n")
        
        md.append(f"Final Score = {base_score:.6f}")
        if acronym_fidelity > 0.0:
            md.append(f" + {acronym_fidelity * 0.15:.6f}")
        if location_score > 0.0:
            md.append(f" + {location_score * 0.05:.6f}")
        md.append(f"\n")
        md.append(f"Final Score = {final_score:.6f} ({final_score*100:.2f}%)\n")
        md.append(f"```\n\n")
        
        # Relative positioning analysis
        md.append("#### Why This Match Ranks at Position #{}\n\n".format(i))
        
        if i == 1:
            md.append(f"**This is the TOP RANKED match** for the query `\"{query}\"`.\n\n")
            md.append(f"It achieved the highest final score of {final_score:.6f} ({final_score*100:.2f}%) among all {len(matches)} matches found.\n\n")
            
            if len(matches) > 1:
                next_match = matches[1]
                next_score = next_match.get('raw_score', next_match.get('score', 0.0))
                score_diff = final_score - next_score
                
                md.append(f"**Comparison with Match #2: \"{next_match['company_name']}\"**\n\n")
                md.append(f"This match ranks above #{2} by a margin of {score_diff:.6f} ({score_diff*100:.2f} percentage points).\n\n")
                
                # Detailed component comparison
                next_string = next_match.get('string_score', 0.0)
                next_sem = next_match.get('normalized_semantic_score', next_match.get('semantic_score', 0.0))
                next_acro = next_match.get('acronym_fidelity', 0.0)
                next_loc = next_match.get('location_score', 0.0)
                
                md.append(f"**Component-by-Component Comparison:**\n\n")
                
                string_diff = string_score - next_string
                if abs(string_diff) > 0.001:
                    md.append(f"- **String Similarity:** {string_score:.6f} vs {next_string:.6f} ")
                    md.append(f"(Δ {string_diff:+.6f}, contributing {string_diff*0.7:+.6f} to score difference)\n")
                    if string_diff > 0:
                        md.append(f"  - This match has BETTER string matching by {abs(string_diff):.6f}, ")
                        md.append(f"which contributes +{string_diff*0.7:.6f} to its advantage.\n")
                    else:
                        md.append(f"  - This match has WORSE string matching by {abs(string_diff):.6f}, ")
                        md.append(f"which reduces its score by {abs(string_diff*0.7):.6f}.\n")
                
                sem_diff = semantic_norm - next_sem
                if abs(sem_diff) > 0.001:
                    md.append(f"- **Semantic Similarity:** {semantic_norm:.6f} vs {next_sem:.6f} ")
                    md.append(f"(Δ {sem_diff:+.6f}, contributing {sem_diff*0.3:+.6f} to score difference)\n")
                    if sem_diff > 0:
                        md.append(f"  - This match has BETTER semantic matching by {abs(sem_diff):.6f}, ")
                        md.append(f"which contributes +{sem_diff*0.3:.6f} to its advantage.\n")
                    else:
                        md.append(f"  - This match has WORSE semantic matching by {abs(sem_diff):.6f}, ")
                        md.append(f"which reduces its score by {abs(sem_diff*0.3):.6f}.\n")
                
                acro_diff = acronym_fidelity - next_acro
                if abs(acro_diff) > 0.001:
                    md.append(f"- **Acronym Fidelity:** {acronym_fidelity:.6f} vs {next_acro:.6f} ")
                    md.append(f"(Δ {acro_diff:+.6f}, contributing {acro_diff*0.15:+.6f} to score difference)\n")
                    if acro_diff > 0:
                        md.append(f"  - This match has BETTER acronym matching by {abs(acro_diff):.6f}, ")
                        md.append(f"which contributes +{acro_diff*0.15:.6f} to its advantage.\n")
                    else:
                        md.append(f"  - This match has WORSE acronym matching by {abs(acro_diff):.6f}, ")
                        md.append(f"which reduces its score by {abs(acro_diff*0.15):.6f}.\n")
                
                md.append(f"\n**Summary:** The cumulative effect of these component differences results in this match ")
                md.append(f"scoring {score_diff:.6f} ({score_diff*100:.2f}%) higher than match #2, ")
                md.append(f"securing its position as the top match.\n\n")
        
        else:
            # Not rank 1
            prev_match = matches[i-2]
            prev_score = prev_match.get('raw_score', prev_match.get('score', 0.0))
            score_diff_above = prev_score - final_score
            
            md.append(f"**Comparison with Match #{i-1} (Ranked Above): \"{prev_match['company_name']}\"**\n\n")
            md.append(f"This match ranks BELOW #{i-1} by a margin of {score_diff_above:.6f} ({score_diff_above*100:.2f} percentage points).\n\n")
            
            # Detailed comparison
            prev_string = prev_match.get('string_score', 0.0)
            prev_sem = prev_match.get('normalized_semantic_score', prev_match.get('semantic_score', 0.0))
            prev_acro = prev_match.get('acronym_fidelity', 0.0)
            
            md.append(f"**Why Match #{i-1} Scores Higher:**\n\n")
            
            string_diff = prev_string - string_score
            if abs(string_diff) > 0.001:
                md.append(f"- **String Similarity Disadvantage:** Match #{i-1} has {string_diff:+.6f} better string score ")
                md.append(f"({prev_string:.6f} vs {string_score:.6f}), contributing {string_diff*0.7:+.6f} to the gap.\n")
            
            sem_diff = prev_sem - semantic_norm
            if abs(sem_diff) > 0.001:
                md.append(f"- **Semantic Similarity Disadvantage:** Match #{i-1} has {sem_diff:+.6f} better semantic score ")
                md.append(f"({prev_sem:.6f} vs {semantic_norm:.6f}), contributing {sem_diff*0.3:+.6f} to the gap.\n")
            
            acro_diff = prev_acro - acronym_fidelity
            if abs(acro_diff) > 0.001:
                md.append(f"- **Acronym Fidelity Disadvantage:** Match #{i-1} has {acro_diff:+.6f} better acronym score ")
                md.append(f"({prev_acro:.6f} vs {acronym_fidelity:.6f}), contributing {acro_diff*0.15:+.6f} to the gap.\n")
            
            md.append(f"\n")
            
            if i < len(matches):
                next_match = matches[i]
                next_score = next_match.get('raw_score', next_match.get('score', 0.0))
                score_diff_below = final_score - next_score
                
                md.append(f"**Comparison with Match #{i+1} (Ranked Below): \"{next_match['company_name']}\"**\n\n")
                md.append(f"This match ranks ABOVE #{i+1} by a margin of {score_diff_below:.6f} ({score_diff_below*100:.2f} percentage points).\n\n")
                
                next_string = next_match.get('string_score', 0.0)
                next_sem = next_match.get('normalized_semantic_score', next_match.get('semantic_score', 0.0))
                next_acro = next_match.get('acronym_fidelity', 0.0)
                
                md.append(f"**Why This Match Scores Higher Than #{i+1}:**\n\n")
                
                string_diff = string_score - next_string
                if abs(string_diff) > 0.001:
                    md.append(f"- **String Similarity Advantage:** This match has {string_diff:+.6f} better string score ")
                    md.append(f"({string_score:.6f} vs {next_string:.6f}), contributing {string_diff*0.7:+.6f} to its lead.\n")
                
                sem_diff = semantic_norm - next_sem
                if abs(sem_diff) > 0.001:
                    md.append(f"- **Semantic Similarity Advantage:** This match has {sem_diff:+.6f} better semantic score ")
                    md.append(f"({semantic_norm:.6f} vs {next_sem:.6f}), contributing {sem_diff*0.3:+.6f} to its lead.\n")
                
                acro_diff = acronym_fidelity - next_acro
                if abs(acro_diff) > 0.001:
                    md.append(f"- **Acronym Fidelity Advantage:** This match has {acro_diff:+.6f} better acronym score ")
                    md.append(f"({acronym_fidelity:.6f} vs {next_acro:.6f}), contributing {acro_diff*0.15:+.6f} to its lead.\n")
                
                md.append(f"\n")
        
        md.append("---\n\n")
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(md))
    
    print(f"📄 Ultra-verbose report saved: {output_file}")


def main():
    print("="*70)
    print("🔬 ULTRA-VERBOSE ANALYSIS - Single Example")
    print("="*70)
    
    # Initialize service
    print("\n📦 Initializing SearchService...")
    service = SearchService()
    if not service.load_company_data():
        print("❌ Failed to load company data")
        return
    
    # Run search for IBM
    query = "IBM"
    print(f"\n🔍 Searching for: {query}")
    matches = service.search(query, top_k=10)
    
    if not matches:
        print("❌ No matches found")
        return
    
    print(f"✅ Found {len(matches)} matches")
    
    # Generate ultra-verbose report
    print("\n📊 Generating ultra-verbose report...")
    generate_ultra_verbose_report(query, matches)
    
    print("\n" + "="*70)
    print("✅ DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
