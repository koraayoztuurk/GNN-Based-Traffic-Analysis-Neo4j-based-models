#!/usr/bin/env python3
"""
BENCHMARK DASHBOARD GENERATOR
==============================

comprehensive_benchmark_results.json dosyasından HTML dashboard oluşturur.

Kullanım:
    python generate_dashboard.py
    python generate_dashboard.py --input results.json --output dashboard.html
"""

import json
import argparse
from datetime import datetime
from pathlib import Path

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Database Benchmark Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 30px;
            border-bottom: 3px solid #667eea;
        }
        
        .header h1 {
            font-size: 3em;
            color: #667eea;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .header .subtitle {
            font-size: 1.2em;
            color: #666;
            font-weight: 300;
        }
        
        .metadata {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            color: white;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .metadata-item {
            text-align: center;
        }
        
        .metadata-label {
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }
        
        .metadata-value {
            font-size: 1.3em;
            font-weight: bold;
        }
        
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .summary-card {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            color: white;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }
        
        .summary-card:hover {
            transform: translateY(-5px);
        }
        
        .summary-card.winner {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }
        
        .summary-title {
            font-size: 1.1em;
            opacity: 0.9;
            margin-bottom: 15px;
        }
        
        .summary-value {
            font-size: 2.5em;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .summary-subtitle {
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 10px;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }
        
        .chart-container {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        
        .chart-title {
            font-size: 1.3em;
            color: #667eea;
            margin-bottom: 20px;
            text-align: center;
            font-weight: 600;
        }
        
        .table-container {
            overflow-x: auto;
            margin-bottom: 40px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        
        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #f0f0f0;
        }
        
        tr:hover {
            background: #f8f9ff;
        }
        
        .winner-badge {
            background: #fee140;
            color: #764ba2;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 30px;
            border-top: 2px solid #e0e0e0;
            color: #666;
            font-size: 0.9em;
        }
        
        @media (max-width: 768px) {
            .charts-grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Database Benchmark Dashboard</h1>
            <div class="subtitle">Comprehensive Performance Analysis</div>
        </div>
        
        <div class="metadata">
            {{METADATA}}
        </div>
        
        <div class="summary">
            {{SUMMARY_CARDS}}
        </div>
        
        <div class="charts-grid">
            {{CHARTS}}
        </div>
        
        <div class="table-container">
            <h2 style="color: #667eea; margin-bottom: 20px;">📋 Detailed Results</h2>
            {{TABLES}}
        </div>
        
        <div class="footer">
            <p>Generated: {{GENERATION_TIME}}</p>
            <p>Traffic Flow Analysis & GNN - Database Benchmark System</p>
        </div>
    </div>
    
    <script>
        // Chart.js configuration
        Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
        Chart.defaults.font.size = 13;
        
        {{CHART_SCRIPTS}}
    </script>
</body>
</html>
"""

def generate_metadata_html(metadata):
    """Generate metadata HTML."""
    html = f"""
        <div class="metadata-item">
            <div class="metadata-label">Timestamp</div>
            <div class="metadata-value">{metadata.get('timestamp', 'N/A')}</div>
        </div>
        <div class="metadata-item">
            <div class="metadata-label">Profile</div>
            <div class="metadata-value">{metadata.get('profile', 'N/A').upper()}</div>
        </div>
        <div class="metadata-item">
            <div class="metadata-label">Databases</div>
            <div class="metadata-value">{', '.join(metadata.get('databases_tested', [])).upper()}</div>
        </div>
    """
    return html

def generate_summary_cards(results):
    """Generate summary cards HTML."""
    html = ""
    
    # Calculate total wins per database
    wins = {}
    total_metrics = 0
    
    for db in results:
        wins[db] = 0
        for test in results[db]:
            for metric in results[db][test]:
                total_metrics += 1
                if results[db][test][metric].get('winner') == db:
                    wins[db] += 1
    
    # Winner card
    winner_db = max(wins, key=wins.get) if wins else "N/A"
    winner_score = wins.get(winner_db, 0)
    
    html += f"""
        <div class="summary-card winner">
            <div class="summary-title">🏆 Overall Winner</div>
            <div class="summary-value">{winner_db.upper()}</div>
            <div class="summary-subtitle">{winner_score}/{total_metrics} metrics won</div>
        </div>
    """
    
    # Individual DB cards
    for db, score in wins.items():
        percentage = (score / total_metrics * 100) if total_metrics > 0 else 0
        html += f"""
            <div class="summary-card">
                <div class="summary-title">{db.upper()}</div>
                <div class="summary-value">{percentage:.1f}%</div>
                <div class="summary-subtitle">{score} metrics won</div>
            </div>
        """
    
    return html

def generate_charts(results):
    """Generate Chart.js charts."""
    charts_html = ""
    scripts = ""
    
    chart_id = 0
    
    # For each database
    for db in results:
        # Collect mean times for each test
        test_names = []
        mean_values = []
        
        for test in sorted(results[db].keys()):
            # Get first metric that has 'Time' or contains time data
            for metric in results[db][test]:
                stats = results[db][test][metric].get('statistics', {})
                if stats and stats.get('mean', 0) > 0:
                    test_names.append(f"{test[:20]}...")
                    mean_values.append(stats['mean'])
                    break
        
        if test_names:
            chart_id += 1
            charts_html += f"""
                <div class="chart-container">
                    <div class="chart-title">{db.upper()} - Mean Response Time</div>
                    <canvas id="chart{chart_id}"></canvas>
                </div>
            """
            
            scripts += f"""
                new Chart(document.getElementById('chart{chart_id}'), {{
                    type: 'bar',
                    data: {{
                        labels: {json.dumps(test_names)},
                        datasets: [{{
                            label: 'Mean Time (ms)',
                            data: {json.dumps(mean_values)},
                            backgroundColor: 'rgba(102, 126, 234, 0.6)',
                            borderColor: 'rgba(102, 126, 234, 1)',
                            borderWidth: 2
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                title: {{
                                    display: true,
                                    text: 'Time (ms)'
                                }}
                            }}
                        }}
                    }}
                }});
            """
    
    # Comparison chart (if multiple databases)
    if len(results) > 1:
        chart_id += 1
        
        # Get common tests
        common_tests = set(results[list(results.keys())[0]].keys())
        for db in results:
            common_tests = common_tests.intersection(set(results[db].keys()))
        
        datasets = []
        colors = [
            'rgba(255, 99, 132, 0.6)',
            'rgba(54, 162, 235, 0.6)',
            'rgba(255, 206, 86, 0.6)',
            'rgba(75, 192, 192, 0.6)'
        ]
        
        for i, db in enumerate(results.keys()):
            values = []
            for test in sorted(common_tests):
                if test in results[db]:
                    for metric in results[db][test]:
                        stats = results[db][test][metric].get('statistics', {})
                        if stats and stats.get('mean', 0) > 0:
                            values.append(stats['mean'])
                            break
                    else:
                        values.append(0)
            
            datasets.append({
                'label': db.upper(),
                'data': values,
                'backgroundColor': colors[i % len(colors)]
            })
        
        test_labels = [test[:20] + '...' for test in sorted(common_tests)]
        
        charts_html += f"""
            <div class="chart-container" style="grid-column: 1 / -1;">
                <div class="chart-title">Database Comparison - Mean Response Time</div>
                <canvas id="chart{chart_id}"></canvas>
            </div>
        """
        
        scripts += f"""
            new Chart(document.getElementById('chart{chart_id}'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(test_labels)},
                    datasets: {json.dumps(datasets)}
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Time (ms)'
                            }}
                        }}
                    }}
                }}
            }});
        """
    
    return charts_html, scripts

def generate_tables(results):
    """Generate detailed results tables."""
    html = ""
    
    for db in results:
        html += f"""
            <h3 style="color: #764ba2; margin-top: 30px; margin-bottom: 15px;">{db.upper()}</h3>
            <table>
                <thead>
                    <tr>
                        <th>Test</th>
                        <th>Metric</th>
                        <th>Mean</th>
                        <th>Median</th>
                        <th>P95</th>
                        <th>P99</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>StdDev</th>
                        <th>Winner</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for test in sorted(results[db].keys()):
            for metric in sorted(results[db][test].keys()):
                stats = results[db][test][metric].get('statistics', {})
                unit = results[db][test][metric].get('unit', '')
                winner = results[db][test][metric].get('winner', '')
                
                winner_badge = f'<span class="winner-badge">WINNER</span>' if winner == db else ''
                
                html += f"""
                    <tr>
                        <td><strong>{test}</strong></td>
                        <td>{metric}</td>
                        <td>{stats.get('mean', 0):.2f} {unit}</td>
                        <td>{stats.get('median', 0):.2f} {unit}</td>
                        <td>{stats.get('p95', 0):.2f} {unit}</td>
                        <td>{stats.get('p99', 0):.2f} {unit}</td>
                        <td>{stats.get('min', 0):.2f} {unit}</td>
                        <td>{stats.get('max', 0):.2f} {unit}</td>
                        <td>{stats.get('std', 0):.2f} {unit}</td>
                        <td>{winner_badge}</td>
                    </tr>
                """
        
        html += """
                </tbody>
            </table>
        """
    
    return html

def generate_dashboard(input_file, output_file):
    """Generate HTML dashboard from benchmark results."""
    
    # Load results
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    results = data.get('results', {})
    
    # Generate components
    metadata_html = generate_metadata_html(metadata)
    summary_html = generate_summary_cards(results)
    charts_html, chart_scripts = generate_charts(results)
    tables_html = generate_tables(results)
    
    # Fill template
    html = HTML_TEMPLATE
    html = html.replace('{{METADATA}}', metadata_html)
    html = html.replace('{{SUMMARY_CARDS}}', summary_html)
    html = html.replace('{{CHARTS}}', charts_html)
    html = html.replace('{{CHART_SCRIPTS}}', chart_scripts)
    html = html.replace('{{TABLES}}', tables_html)
    html = html.replace('{{GENERATION_TIME}}', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n[OK] Dashboard created: {output_file}")
    print(f"     Open in browser: file:///{Path(output_file).absolute()}")

def main():
    # Proje root dizinini belirle
    ROOT_DIR = Path(__file__).parent.parent.parent
    
    parser = argparse.ArgumentParser(description="Generate HTML dashboard from benchmark results")
    parser.add_argument(
        '--input', 
        default=str(ROOT_DIR / 'outputs' / 'benchmarks' / 'comprehensive_benchmark_results.json'), 
        help='Input JSON file'
    )
    parser.add_argument(
        '--output', 
        default=str(ROOT_DIR / 'outputs' / 'benchmarks' / 'benchmark_dashboard.html'), 
        help='Output HTML file'
    )
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"[ERROR] Input file not found: {args.input}")
        print(f"        Run benchmark first: python tools/benchmark/runner.py --profile standard")
        return
    
    generate_dashboard(args.input, args.output)

if __name__ == "__main__":
    main()
