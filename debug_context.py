
def analyze_business_context(query, company_name):
    business_keywords = {
        'corporate': ['corp', 'corporation', 'incorporated', 'inc', 'llc', 'ltd', 'limited'],
        'partnership': ['partners', 'partnership', 'associates', 'assoc'],
        'holding': ['holdings', 'holding', 'group', 'enterprises', 'ventures'],
        'international': ['intl', 'international', 'global', 'worldwide'],
        'regional': ['regional', 'national', 'local', 'state', 'city'],
        'technology': ['tech', 'technology', 'digital', 'software', 'systems'],
        'financial': ['financial', 'finance', 'capital', 'investment', 'funds'],
        'consulting': ['consulting', 'consultants', 'advisory', 'services'],
        'religious': ['church', 'synagogue', 'temple', 'ministry', 'messianic', 'assemblies', 'catholic'],
        'non_profit': ['association', 'foundation', 'club', 'society', 'coalition', 'initiative', 'center', 'charity']
    }
    
    query = query.lower()
    company_name = company_name.lower()
    
    print(f"Analyzing Query: '{query}'")
    for business_type, keywords in business_keywords.items():
        for keyword in keywords:
            if keyword in query:
                print(f"  MATCH: type='{business_type}', keyword='{keyword}'")

    print(f"Analyzing Company: '{company_name}'")
    for business_type, keywords in business_keywords.items():
        for keyword in keywords:
            if keyword in company_name:
                print(f"  MATCH: type='{business_type}', keyword='{keyword}'")

analyze_business_context("Kehilat Ariel Synagogue", "Kehilat Ariel Synagogue (San Diego, CA)")
