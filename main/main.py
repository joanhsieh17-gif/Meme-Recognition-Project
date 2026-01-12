import os
import json
import argparse
import base64
import mimetypes
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from MemeAnalyzer import MemeAnalyzer

def generate_html_report(results, report_path):
    """Generate a clean HTML report showing images and analysis results."""
    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Meme Analysis Report - {datetime.now().strftime("%Y-%m-%d")}</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background-color: #f9f9f9; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; background-color: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: top; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            img {{ max-width: 300px; height: auto; border-radius: 4px; display: block; margin-bottom: 5px; }}
            .path {{ font-size: 0.8em; color: #666; word-break: break-all; }}
            .reason {{ font-style: italic; color: #444; }}
            .names {{ font-weight: bold; color: #2c3e50; }}
        </style>
    </head>
    <body>
        <h1>Meme Analysis Report</h1>
        <p>Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <table>
            <thead>
                <tr>
                    <th>Meme Image</th>
                    <th>Face Recognition Candidates</th>
                    <th>OCR Extracted Text</th>
                    <th>Final Prediction</th>
                    <th>Explanation</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for res in results:
        img_path_str = res['meme_path']
        img_src = ""
        
        # Try to embed image as base64
        try:
            img_path = Path(img_path_str)
            if img_path.exists():
                mime_type, _ = mimetypes.guess_type(img_path)
                if not mime_type:
                    mime_type = "image/jpeg" # Default fallback
                
                with open(img_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    img_src = f"data:{mime_type};base64,{encoded_string}"
            else:
                img_src = "" # File not found placeholder or empty
        except Exception as e:
            print(f"Warning: Could not embed image {img_path_str}: {e}")
            img_src = ""

        # If embedding failed, maybe fallback to path (though likely won't work in browser if local)
        if not img_src:
             img_src = img_path_str

        candidates = ", ".join(res['candidates']) if res['candidates'] else "None"
        ocr = "<br>".join(res['ocr_texts'][:15]) if res['ocr_texts'] else "None"
        if len(res['ocr_texts']) > 15: ocr += "<br>..."
        
        names_zh = res['llm_pred_names']
        names_py = res['llm_pred_names_pinyin']
        
        pred_display = []
        for zh, py in zip(names_zh, names_py):
            pred_display.append(f"{zh} ({py})")
        
        prediction = "<br>".join(pred_display) if pred_display else "None"
        reason = res['llm_reason']
        
        html_content += f"""
                <tr>
                    <td>
                        <img src="{img_src}" alt="Meme">
                        <div class="path">{img_path_str}</div>
                    </td>
                    <td>{candidates}</td>
                    <td>{ocr}</td>
                    <td class="names">{prediction}</td>
                    <td class="reason">{reason}</td>
                </tr>
        """

    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"\nHTML report successfully saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch Meme Analysis Tool")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing images to analyze")
    parser.add_argument("--model_path", type=str, default="../models/trained_svm_model_real25.pkl", help="Path to SVM model")
    parser.add_argument("--api_key", type=str, help="Google Gemini API Key (Optional if env var set)")
    
    args = parser.parse_args()
    
    # 1. Initialize Analyzer
    try:
        analyzer = MemeAnalyzer(
            face_model_path=args.model_path, 
            gemini_api_key=args.api_key
        )
    except Exception as e:
        print(f"Error: Failed to initialize MemeAnalyzer: {e}")
        return

    # 2. Prepare output directories
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path("results")
    reports_dir = Path("reports")
    results_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    
    output_jsonl = results_dir / f"{timestamp}.jsonl"
    output_html = reports_dir / f"{timestamp}.html"
    
    # 3. Find images
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"Error: Test directory '{args.test_dir}' does not exist.")
        return

    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = [f for f in test_dir.iterdir() if f.suffix.lower() in image_extensions]
    image_files.sort()
    
    if not image_files:
        print(f"No valid images found in {args.test_dir}")
        return
        
    print(f"Starting analysis of {len(image_files)} images...")
    
    results = []
    
    # 4. Run Analysis
    with open(output_jsonl, 'w', encoding='utf-8') as f_jsonl:
        for img_file in tqdm(image_files, desc="Batch Processing"):
            try:
                # Perform analysis
                res_dict = analyzer.analyze(str(img_file))
                
                # Add file path to result
                full_res = {
                    "meme_path": str(img_file),
                    **res_dict
                }
                
                # Save to JSONL 
                f_jsonl.write(json.dumps(full_res, ensure_ascii=False) + "\n")
                f_jsonl.flush()
                
                results.append(full_res)
                
            except Exception as e:
                print(f"\nFailed to analyze {img_file.name}: {e}")
    
    # 5. Generate HTML Report
    if results:
        generate_html_report(results, output_html)
        print(f"Analysis complete. JSONL saved to: {output_jsonl}")
    else:
        print("No images were successfully analyzed.")

if __name__ == "__main__":
    main()
