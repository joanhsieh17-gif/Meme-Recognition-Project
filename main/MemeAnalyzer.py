import os
import json
import pickle
import numpy as np
import face_recognition
import hashlib
from pathlib import Path
from PIL import Image
from paddleocr import PaddleOCR
from google import genai
from pypinyin import pinyin, Style
import argparse

# --- Configuration Constants ---
POLITICIAN_LIST = [
    "賴清德", "曹興誠", "柯建銘", "林智堅", "郭昱晴 (萬老師)", "陳其邁", "邱議瑩",
    "王義川", "陳吉仲", "沈伯洋", "吳崢", "李進勇", "林楚茵", "呂建德", "林俊憲", "賴品妤",
    "吳思瑤", "李俊俋", "蔡英文", "吳靜怡", "黃捷", "蔡其昌", "吳沛憶", "劉世芳", "王定宇", 
    "卓榮泰", "黃偉哲"
]

PROMPT = """
你是一位專精於臺灣政治迷因的分析專家，擅長解讀圖片中的視覺隱喻、政治反諷及時事梗。
請根據提供的圖片資訊、OCR 文字及參考名單，完成以下分析任務。

# 任務說明
1. **人物識別**：判斷迷因圖中是否出現（視覺人臉）或文字提及（OCR 內容）`politician_list` 中的政治人物。
2. **內容解讀**：以繁體中文撰寫一句話，精簡說明迷因內容（包含諷刺議題、政治背景或人物行為）。

# 輸入資料
- **meme_texts**（OCR 文字）：{texts}
- **possible_names**（參考線索）：{names} （註：此為人臉辨識模型的初步結果，準確度高但可能包含列表外的人物，請以此為重要線索並搭配 politician_list 過濾）
- **politician_list**（允許的候選名單）：{politician_list}

# 輸出規則 (嚴格遵守)
1. **pred_names (List[str])**
   - **封閉選項**：結果必須完全來自 `politician_list`。絕對不可自行創造、翻譯或使用列表以外的名字。
   - **判斷邏輯**：
     - 請綜合考量 `possible_names` (人臉線索) 與 `meme_texts` (文字線索)。
     - 若 `possible_names` 中的名字也在 `politician_list` 中，請優先納入。
     - 若圖中人物有綽號（例如 OCR 出現「柯P」、「小英」），請自動對應回 `politician_list` 中的本名（如：柯文哲、蔡英文）。
   - **空值處理**：若迷因中未出現或提及名單內的任何人物，請回傳空列表 `[]`（不要強行預測）。

2. **reason (str)**
   - **單一句子**：必須是語意完整的一句話。
   - **內容焦點**：請指出「誰」在「什麼議題」上被「如何描繪/諷刺」。
   - **語言**：繁體中文。

3. **格式限制**
   - 僅輸出標準 JSON 格式，不要包含 Markdown 標記（如 ```json ... ```）或任何額外說明的文字。

# 輸出範例
{{
  "pred_names": ["林智堅", "蔡英文"],
  "reason": "此圖諷刺林智堅在論文案爭議中，獲得黨內大力的支持與背書。"
}}
"""

class MemeAnalyzer:
    def __init__(self, 
                 face_model_path: str, 
                 gemini_api_key: str = None,
                 recg_threshold: float = 0.1, 
                 ocr_threshold: float = 0.9,
                 ocr_cache_dir: str = "caches/ocr_cache"):
        
        print("Initializing MemeAnalyzer...")
        
        # 1. Init Gemini
        self.api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
             raise ValueError("CRITICAL: Gemini API Key not found! Set GEMINI_API_KEY env var or pass it in.")
             
        self.client = genai.Client(api_key=self.api_key)
        
        # 2. Init Face Recognition Model
        print(f"Loading face model from {face_model_path}...")
        try:
            with open(face_model_path, "rb") as f:
                self.face_model = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load face model: {e}")
            
        self.recg_threshold = recg_threshold
        
        # 3. Init PaddleOCR
        print("Loading PaddleOCR...")
        self.ocr = PaddleOCR()
        self.ocr_threshold = ocr_threshold
        
        # 4. Cache Settings
        self.ocr_cache_dir = Path(ocr_cache_dir)
        self.ocr_cache_dir.mkdir(parents=True, exist_ok=True)
        
        print("MemeAnalyzer ready!")

    def _get_image_array(self, image_file):
        """Convert image file to numpy array and PIL Image"""
        img = Image.open(image_file).convert("RGB")
        return np.array(img), img

    def _predict_faces(self, img_array):
        """Recognize faces in the image and return candidates with details"""
        # Get face locations
        face_locations = face_recognition.face_locations(img_array)
        if not face_locations:
            return [], []

        # Get face encodings
        encodings = face_recognition.face_encodings(img_array, known_face_locations=face_locations)
        
        # Get candidates
        candidates = set()
        cand_details = []
        
        for idx, encoding in enumerate(encodings):
            probabilities = self.face_model.predict_proba([encoding])[0]
            # Find the best match for this face
            best_idx = np.argmax(probabilities)
            best_prob = probabilities[best_idx]
            best_name = self.face_model.classes_[best_idx]
            
            # Record detailed info for the top prediction of this face
            cand_details.append({
                "face_index": idx + 1,
                "name": best_name,
                "prob": float(f"{best_prob:.4f}")
            })

            # Check against threshold for the final set
            for name, prob in zip(self.face_model.classes_, probabilities):
                if prob >= self.recg_threshold:
                    candidates.add(name)
        
        return list(candidates), cand_details

    def _extract_ocr(self, img_array, image_path: str = None):
        """Extract OCR text from the image with caching support (returns text list and details)"""
        
        cache_file = None
        if image_path:
            # Create an unique key for the image based on its absolute path
            abs_path = os.path.abspath(image_path)
            key_source = f"{abs_path}|{self.ocr_threshold}"
            cache_name = hashlib.md5(key_source.encode("utf-8")).hexdigest()
            cache_file = self.ocr_cache_dir / f"{cache_name}_v2.json" # v2 for new format
            
            # If cache exists, return cached results
            if cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("texts", []), data.get("details", [])

        # If no cache, perform OCR
        result = self.ocr.predict(input=img_array)
        texts = []
        details = []
        
        if result and len(result) > 0:
            # Handle PaddleX OCRResult (dict-like)
            if hasattr(result[0], 'keys') and 'rec_texts' in result[0]:
                 rec_texts = result[0]['rec_texts']
                 rec_scores = result[0]['rec_scores']
                 for text, score in zip(rec_texts, rec_scores):
                      if score >= self.ocr_threshold:
                           texts.append(text)
                           details.append({"text": text, "prob": float(f"{score:.4f}")})
            
            # Handle legacy PaddleOCR format (list of lists)
            elif isinstance(result[0], list):
                for line in result[0]:
                    # line structure: [box_points, [text, score]]
                    if len(line) >= 2 and isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                        text = line[1][0]
                        confidence = line[1][1]
                        if confidence >= self.ocr_threshold:
                            texts.append(text)
                            details.append({"text": text, "prob": float(f"{confidence:.4f}")})
        
        # Save to cache if path provided
        if cache_file:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"texts": texts, "details": details}, f, ensure_ascii=False)
                
        return texts, details

    def _get_pinyin(self, chinese_text):
        """Convert Chinese text to pinyin string"""
        res = pinyin(chinese_text, style=Style.NORMAL)
        return '_'.join([''.join(p) for p in res])

    def _call_gemini(self, candidates, ocr_texts, img_pil):
        """Call Gemini to analyze the image"""
        # Format prompt
        prompt = PROMPT.format(
            names=candidates,
            texts=ocr_texts,
            politician_list=POLITICIAN_LIST
        )
        
        try:
            # Call Gemini
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[img_pil, prompt]
            )
            # Process the response
            cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_text)
        except Exception as e:
            print(f"Gemini Error: {e}")
            return {"pred_names": [], "reason": "無法分析圖片內容"}
    
    def _parse_filename_info(self, image_path):
        """Extract meme_num and meme_names from filename"""
        filename = os.path.basename(image_path)
        stem = Path(filename).stem
        
        parts = stem.split('_', 1)
        if len(parts) > 1 and parts[0].isdigit():
            meme_num = parts[0]
            name_part = parts[1]
        else:
            meme_num = "000"
            name_part = stem
            
        meme_names = name_part.split('&')
        return meme_num, meme_names

    def analyze(self, image_path):
        """Main analysis pipeline"""
        # Get image array and PIL image
        img_array, img_pil = self._get_image_array(image_path)
        
        # Predict faces and extract OCR text
        candidates, cand_details = self._predict_faces(img_array)
        ocr_texts, ocr_details = self._extract_ocr(img_array, image_path)
        
        print(f"Faces (Candidates): {candidates}")
        print(f"OCR: {ocr_texts[:3]}...")
        
        # Call Gemini
        result = self._call_gemini(candidates, ocr_texts, img_pil)
        names_zh = result.get('pred_names', [])
        reason = result.get('reason', 'No analysis result')
        
        # Convert names to pinyin
        names_pinyin = [self._get_pinyin(name) for name in names_zh]
        
        # Parse filename info
        meme_num, meme_names = self._parse_filename_info(image_path)
        
        return {
            "candidates": candidates,
            "cand_details": cand_details,
            "meme_num": meme_num,
            "meme_names": meme_names,
            "ocr_texts": ocr_texts,
            "ocr_details": ocr_details,
            "llm_pred_names": names_zh,
            "llm_pred_names_pinyin": names_pinyin,
            "llm_reason": reason
        }