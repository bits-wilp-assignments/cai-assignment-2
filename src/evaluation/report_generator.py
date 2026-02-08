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
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #2c3e50;
            background-color: #ffffff;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #0078d4 0%, #00a2ed 100%);
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 4px 6px rgba(0, 120, 212, 0.15);
        }}
        .header h1 {{
            font-size: 2em;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 15px;
        }}
        .header-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        .header-info-item {{
            font-size: 0.95em;
            color: #e6f4ff;
        }}
        .header-info-item strong {{
            color: #ffffff;
            font-weight: 500;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            font-size: 1.5em;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e0e0e0;
        }}
        .section h3 {{
            font-size: 1.2em;
            font-weight: 500;
            color: #34495e;
            margin: 25px 0 15px 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            text-align: center;
        }}
        .metric-card .label {{
            font-size: 0.85em;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
            font-weight: 500;
        }}
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: 600;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            border: 1px solid #e0e0e0;
        }}
        th {{
            background-color: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #2c3e50;
            border-bottom: 2px solid #e0e0e0;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .good {{ color: #27ae60; font-weight: 600; }}
        .medium {{ color: #f39c12; font-weight: 600; }}
        .poor {{ color: #e74c3c; font-weight: 600; }}
        .info-box {{
            background-color: #f8f9fa;
            border-left: 3px solid #3498db;
            padding: 20px;
            margin: 20px 0;
        }}
        .info-box h4 {{
            font-size: 1em;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 12px;
        }}
        .info-box ul {{
            list-style: none;
            padding-left: 0;
        }}
        .info-box li {{
            padding: 6px 0;
            color: #555;
            font-size: 0.95em;
        }}
        .info-box li strong {{
            color: #2c3e50;
            font-weight: 500;
        }}
        .footer {{
            text-align: center;
            color: #7f8c8d;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RAG System Evaluation Report</h1>
            <div class="header-info">
                <div class="header-info-item"><strong>Generated:</strong> {timestamp}</div>
                <div class="header-info-item"><strong>Dataset:</strong> {os.path.basename(dataset_file)}</div>
                <div class="header-info-item"><strong>Total Questions:</strong> {summary['total_questions']}</div>
                <div class="header-info-item"><strong>Mode:</strong> {'Answer Quality Only' if answer_only_mode else 'Full Evaluation'}</div>
            </div>
        </div>
"""

    # Overall Metrics Section
    html_content += """
        <div class="section">
            <h2>Overall Performance Metrics</h2>
"""

    if not answer_only_mode and summary.get('overall_mrr') is not None:
        mrr_value = summary['overall_mrr']
        mrr_class = 'good' if mrr_value >= 0.7 else 'medium' if mrr_value >= 0.5 else 'poor'
        retrieval_exec_count = summary.get('questions_evaluated_retrieval', 0)
        html_content += f"""
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="label">Mean Reciprocal Rank</div>
                    <div class="value {mrr_class}">{mrr_value:.4f}</div>
                    <div style="font-size: 0.8em; color: #7f8c8d; margin-top: 8px;">({retrieval_exec_count} executions)</div>
                </div>
            </div>
"""

    # Answer Quality Metrics
    if summary.get('answer_evaluation_enabled'):
        answer_exec_count = summary.get('questions_evaluated_answer', 0)
        html_content += f"""
            <h3>Answer Quality Metrics</h3>
            <div style="font-size: 0.9em; color: #555; margin-bottom: 10px;"><strong>Executions:</strong> {answer_exec_count} questions evaluated</div>
            <div class="metrics-grid">
"""
        answer_metrics = [
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
                <div class="metric-card">
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
        retrieval_exec_count = summary.get('questions_evaluated_retrieval', 0)
        html_content += f"""
            <h3>Retrieval Performance</h3>
            <div style="font-size: 0.9em; color: #555; margin-bottom: 10px;"><strong>Executions:</strong> {retrieval_exec_count} questions evaluated</div>
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
            html_content += f'                    <td class="{val_class}">{val:.4f}</td>'

        html_content += """
                </tr>
                <tr>
                    <td><strong>Recall</strong></td>
"""
        for k in [3, 5, 10]:
            val = overall[f'recall_at_{k}']
            val_class = 'good' if val >= 0.7 else 'medium' if val >= 0.5 else 'poor'
            html_content += f'                    <td class="{val_class}">{val:.4f}</td>'

        html_content += """
                </tr>
                <tr>
                    <td><strong>Hit Rate</strong></td>
"""
        for k in [3, 5, 10]:
            val = overall[f'hit_rate_at_{k}']
            val_class = 'good' if val >= 0.7 else 'medium' if val >= 0.5 else 'poor'
            html_content += f'                    <td class="{val_class}">{val:.4f}</td>'

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
            <h2>Performance by Question Type</h2>
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
            if 'f1_score' in metrics:
                type_answer_metrics = [
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
                <div class="metric-card">
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
            <h2>Metrics Interpretation</h2>
            <div class="info-box">
                <h4>Answer Quality Metrics:</h4>
                <ul>
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
                    <li><strong>MRR:</strong> Mean Reciprocal Rank - average of reciprocal ranks of first relevant result</li>
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
        </div>
    </div>
</body>
</html>
"""

    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_path
