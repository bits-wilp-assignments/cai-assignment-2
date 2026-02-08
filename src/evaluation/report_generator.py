"""
Report Generator for RAG Evaluation Results
Generates HTML reports from evaluation results
"""
import os
from datetime import datetime


def generate_html_report(results: dict, dataset_file: str, output_path: str):
    """Generate HTML evaluation report.
    
    Args:
        results: Evaluation results dictionary
        dataset_file: Path to the dataset file used for evaluation
        output_path: Path where the HTML report will be saved
        
    Returns:
        str: Path to the generated HTML report
    """
    summary = results['summary']
    answer_only_mode = summary.get('answer_only_mode', False)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG System Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 5px 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-card .label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .answer-metrics {{
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .good {{ color: #27ae60; font-weight: bold; }}
        .medium {{ color: #f39c12; font-weight: bold; }}
        .poor {{ color: #e74c3c; font-weight: bold; }}
        .info-box {{
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            padding: 20px;
            border-top: 2px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 RAG System Evaluation Report</h1>
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>Dataset:</strong> {os.path.basename(dataset_file)}</p>
        <p><strong>Total Questions:</strong> {summary['total_questions']}</p>
        <p><strong>Evaluation Mode:</strong> {'Answer Quality Only' if answer_only_mode else 'Full Evaluation (Retrieval + Answer Quality)'}</p>
    </div>
"""

    # Overall Metrics Section
    html_content += """
    <div class="section">
        <h2>📊 Overall Performance Metrics</h2>
"""

    if not answer_only_mode and summary.get('overall_mrr') is not None:
        mrr_value = summary['overall_mrr']
        mrr_class = 'good' if mrr_value >= 0.7 else 'medium' if mrr_value >= 0.5 else 'poor'
        html_content += f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">Mean Reciprocal Rank</div>
                <div class="value {mrr_class}">{mrr_value:.4f}</div>
            </div>
        </div>
"""

    # Answer Quality Metrics
    if summary.get('answer_evaluation_enabled'):
        html_content += """
        <h3>Answer Quality Metrics</h3>
        <div class="metrics-grid">
"""
        answer_metrics = [
            ('overall_exact_match', 'Exact Match (EM)'),
            ('overall_f1', 'F1 Score'),
            ('overall_bleu', 'BLEU Score'),
            ('overall_rouge_l', 'ROUGE-L'),
            ('overall_semantic_similarity', 'Semantic Similarity')
        ]
        
        for key, label in answer_metrics:
            if summary.get(key) is not None:
                value = summary[key]
                value_class = 'good' if value >= 0.7 else 'medium' if value >= 0.5 else 'poor'
                html_content += f"""
            <div class="metric-card answer-metrics">
                <div class="label">{label}</div>
                <div class="value {value_class}">{value:.4f}</div>
            </div>
"""
        html_content += """
        </div>
"""

    # Retrieval Metrics Table
    if 'overall_metrics' in summary and not answer_only_mode:
        overall = summary['overall_metrics']
        html_content += """
        <h3>Retrieval Performance</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>@3</th>
                <th>@5</th>
                <th>@10</th>
            </tr>
            <tr>
                <td><strong>Precision</strong></td>
"""
        for k in [3, 5, 10]:
            val = overall[f'precision_at_{k}']
            val_class = 'good' if val >= 0.7 else 'medium' if val >= 0.5 else 'poor'
            html_content += f'<td class="{val_class}">{val:.4f}</td>'
        
        html_content += """
            </tr>
            <tr>
                <td><strong>Recall</strong></td>
"""
        for k in [3, 5, 10]:
            val = overall[f'recall_at_{k}']
            val_class = 'good' if val >= 0.7 else 'medium' if val >= 0.5 else 'poor'
            html_content += f'<td class="{val_class}">{val:.4f}</td>'
        
        html_content += """
            </tr>
            <tr>
                <td><strong>Hit Rate</strong></td>
"""
        for k in [3, 5, 10]:
            val = overall[f'hit_rate_at_{k}']
            val_class = 'good' if val >= 0.7 else 'medium' if val >= 0.5 else 'poor'
            html_content += f'<td class="{val_class}">{val:.4f}</td>'
        
        html_content += """
            </tr>
        </table>
"""

    html_content += """
    </div>
"""

    # Per-Question-Type Metrics
    if summary.get('metrics_by_question_type'):
        html_content += """
    <div class="section">
        <h2>📋 Performance by Question Type</h2>
"""
        for qtype, metrics in summary['metrics_by_question_type'].items():
            html_content += f"""
        <h3>{qtype.upper()} ({metrics['count']} questions)</h3>
        <div class="metrics-grid">
"""
            if not answer_only_mode and 'mrr' in metrics:
                mrr_val = metrics['mrr']
                mrr_class = 'good' if mrr_val >= 0.7 else 'medium' if mrr_val >= 0.5 else 'poor'
                html_content += f"""
            <div class="metric-card">
                <div class="label">MRR</div>
                <div class="value {mrr_class}">{mrr_val:.4f}</div>
            </div>
"""
            
            # Answer quality metrics for this type
            if 'exact_match' in metrics:
                type_answer_metrics = [
                    ('exact_match', 'EM'),
                    ('f1_score', 'F1'),
                    ('bleu_score', 'BLEU'),
                    ('rouge_l', 'ROUGE-L'),
                    ('semantic_similarity', 'Semantic')
                ]
                
                for key, label in type_answer_metrics:
                    if key in metrics:
                        val = metrics[key]
                        val_class = 'good' if val >= 0.7 else 'medium' if val >= 0.5 else 'poor'
                        html_content += f"""
            <div class="metric-card answer-metrics">
                <div class="label">{label}</div>
                <div class="value {val_class}">{val:.4f}</div>
            </div>
"""
            
            html_content += """
        </div>
"""

        html_content += """
    </div>
"""

    # Interpretation Guide
    html_content += """
    <div class="section">
        <h2>📖 Metrics Interpretation Guide</h2>
        <div class="info-box">
            <h4>Answer Quality Metrics:</h4>
            <ul>
                <li><strong>Exact Match (EM):</strong> Percentage of predictions that match ground truth exactly (after normalization)</li>
                <li><strong>F1 Score:</strong> Harmonic mean of precision and recall at token level</li>
                <li><strong>BLEU:</strong> Measures n-gram overlap between prediction and reference</li>
                <li><strong>ROUGE-L:</strong> Longest common subsequence based metric</li>
                <li><strong>Semantic Similarity:</strong> Cosine similarity between answer embeddings</li>
            </ul>
        </div>
"""

    if not answer_only_mode:
        html_content += """
        <div class="info-box">
            <h4>Retrieval Metrics:</h4>
            <ul>
                <li><strong>MRR (Mean Reciprocal Rank):</strong> Average of reciprocal ranks of first relevant result</li>
                <li><strong>Precision@K:</strong> Proportion of relevant documents in top K results</li>
                <li><strong>Recall@K:</strong> Proportion of all relevant documents found in top K</li>
                <li><strong>Hit Rate@K:</strong> Percentage of queries with at least one relevant result in top K</li>
            </ul>
        </div>
"""

    html_content += """
        <div class="info-box">
            <h4>Score Interpretation:</h4>
            <ul>
                <li><span class="good">≥ 0.70:</span> Good performance</li>
                <li><span class="medium">0.50 - 0.69:</span> Moderate performance</li>
                <li><span class="poor">< 0.50:</span> Needs improvement</li>
            </ul>
        </div>
    </div>
"""

    # Footer
    html_content += f"""
    <div class="footer">
        <p>Generated by RAG Evaluation System</p>
        <p>Report saved to: {output_path}</p>
    </div>
</body>
</html>
"""

    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path
