import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

def fetch_arxiv_papers(query, max_results=30):
    url = f"http://export.arxiv.org/api/query?search_query=all:%22{urllib.parse.quote(query)}%22&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        data = response.read()
        root = ET.fromstring(data)
        
        papers = []
        entries = root.findall('{http://www.w3.org/2005/Atom}entry')
        for entry in entries:
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.replace('\n', ' ').strip()
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.replace('\n', ' ').strip()
            papers.append({"title": title, "summary": summary})
        return papers
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

if __name__ == "__main__":
    queries = ["TabPFN", "in-context learning tabular", "TabICL", "TabFM"]
    all_papers = []
    
    print("Gathering data from ArXiv for Tabular Foundation Models...")
    for q in queries:
        papers = fetch_arxiv_papers(q, max_results=15)
        all_papers.extend(papers)
        
    print(f"\nTotal papers gathered: {len(all_papers)}")
    
    # Analyze common themes
    themes = {
        "Efficiency/Speed": ["fast", "efficient", "scaling", "distillation", "memory", "latency"],
        "Robustness/Adversarial": ["robust", "adversarial", "attack", "noise", "out-of-distribution"],
        "Interpretability": ["interpretable", "explainable", "attention", "shap", "feature importance", "causal"],
        "Federated/Privacy": ["federated", "privacy", "differential privacy", "secure"],
        "Domain Specific": ["medical", "finance", "healthcare", "biology", "genomics"],
        "Imputation/Missing Data": ["imputation", "missing", "incomplete"],
    }
    
    theme_counts = {t: 0 for t in themes}
    for p in all_papers:
        text = (p['title'] + " " + p['summary']).lower()
        for t, keywords in themes.items():
            if any(kw in text for kw in keywords):
                theme_counts[t] += 1
                
    print("\n--- Research Themes Identified ---")
    for t, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{t}: {count} papers")
