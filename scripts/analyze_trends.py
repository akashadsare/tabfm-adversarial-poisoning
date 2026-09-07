import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import re
from collections import Counter

def fetch_papers(query, max_results=50):
    url = f"http://export.arxiv.org/api/query?search_query=all:%22{urllib.parse.quote(query)}%22&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=ascending"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        data = response.read()
        root = ET.fromstring(data)
        
        titles = []
        abstracts = []
        entries = root.findall('{http://www.w3.org/2005/Atom}entry')
        for entry in entries:
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.replace('\n', ' ').strip()
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.replace('\n', ' ').strip()
            titles.append(title)
            abstracts.append(summary)
        return titles, abstracts
    except Exception as e:
        print(f"Error fetching data: {e}")
        return [], []

def analyze_trends(titles, abstracts):
    # Combine all text
    text = " ".join(titles).lower()
    
    # Common research direction keywords
    categories = {
        "Efficiency & Scaling": ["efficient", "scaling", "distillation", "quantization", "fast", "memory", "scalable"],
        "Robustness & Adversarial": ["robust", "adversarial", "attack", "defense", "noise", "uncertainty"],
        "Domain Adaptation & Transfer": ["medical", "finance", "healthcare", "domain", "transfer", "adaptation", "specific"],
        "Interpretability & Explainability": ["interpretable", "explainable", "understanding", "attention", "feature importance", "shap"],
        "Benchmarking & Evaluation": ["benchmark", "evaluation", "empirical", "revisiting", "comparing", "dataset"],
        "Methodological Extensions": ["multimodal", "generative", "imputation", "time-series", "graphs", "integration"]
    }
    
    counts = {cat: 0 for cat in categories}
    
    for title in titles:
        title_lower = title.lower()
        for cat, keywords in categories.items():
            if any(kw in title_lower for kw in keywords):
                counts[cat] += 1
                
    return counts, titles

if __name__ == "__main__":
    print("Analyzing follow-up research trends for 'TabPFN' (a predecessor tabular foundation model)...")
    titles, abstracts = fetch_papers("TabPFN", max_results=40)
    
    counts, titles = analyze_trends(titles, abstracts)
    
    print("\n--- Research Trend Categories for Tabular Foundation Models ---")
    for cat, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{cat}: {count} papers")
        
    print("\n--- Sample of Actual Follow-up Paper Titles ---")
    for t in titles[:10]:
        print(f"- {t}")
