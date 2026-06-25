import json
import os
import glob
from collections import Counter

def analyze_file(filepath):
    stats = {
        'count': 0,
        'has_hallucinations': 0,
        'no_hallucinations': 0,
        'total_labels': 0,
        'categories': Counter(),
        'probabilities': Counter(),
        'response_lens': []
    }
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            stats['count'] += 1
            stats['response_lens'].append(len(data['response']))
            
            labels = data.get('labels', [])
            if len(labels) > 0:
                stats['has_hallucinations'] += 1
                stats['total_labels'] += len(labels)
                for label in labels:
                    stats['categories'][label['label']] += 1
                    # Round float representation to avoid float issues
                    prob = round(label['prob'], 4)
                    stats['probabilities'][prob] += 1
            else:
                stats['no_hallucinations'] += 1
                
    return stats

def main():
    data_dir = r"c:\Users\geama\Documents\research\Shroom\shroom-visions-data\distrib"
    pattern = os.path.join(data_dir, "shroom-vision.train.*.labeled.jsonl")
    files = glob.glob(pattern)
    
    print("=" * 80)
    print(f"{'Language':<10} | {'Total':<8} | {'With Halluc':<12} | {'No Halluc':<10} | {'Avg Len':<8} | {'Total Labels':<12}")
    print("-" * 80)
    
    all_categories = Counter()
    all_probs = Counter()
    total_samples = 0
    total_with = 0
    total_no = 0
    
    for fp in sorted(files):
        lang = os.path.basename(fp).split('.')[2]
        stats = analyze_file(fp)
        
        total_samples += stats['count']
        total_with += stats['has_hallucinations']
        total_no += stats['no_hallucinations']
        all_categories.update(stats['categories'])
        all_probs.update(stats['probabilities'])
        
        avg_len = sum(stats['response_lens']) / len(stats['response_lens']) if stats['response_lens'] else 0
        
        print(f"{lang:<10} | {stats['count']:<8} | {stats['has_hallucinations']:<12} | {stats['no_hallucinations']:<10} | {avg_len:<8.1f} | {stats['total_labels']:<12}")
        
    print("-" * 80)
    print(f"{'TOTAL':<10} | {total_samples:<8} | {total_with:<12} | {total_no:<10} | {'-':<8} | {sum(all_categories.values()):<12}")
    print("=" * 80)
    
    print("\nCategory Distribution (across all languages):")
    for cat, count in all_categories.most_common():
        percentage = (count / sum(all_categories.values())) * 100 if all_categories.values() else 0
        print(f"  - {cat:<20}: {count:<6} ({percentage:.1f}%)")
        
    print("\nProbability (Annotator Agreement) Distribution:")
    for prob in sorted(all_probs.keys()):
        count = all_probs[prob]
        percentage = (count / sum(all_probs.values())) * 100 if all_probs.values() else 0
        print(f"  - Prob {prob:.3f} (approx {int(prob*3)}/3 votes): {count:<6} ({percentage:.1f}%)")

if __name__ == '__main__':
    main()
