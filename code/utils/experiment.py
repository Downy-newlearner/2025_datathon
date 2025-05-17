import os
import time
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
import requests

from code.config import ExperimentConfig
from code.prompts.templates import TEMPLATES
from code.utils.metrics import evaluate_correction

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        self.template = TEMPLATES[config.template_name]
        self.api_url = config.api_url
        self.model = config.model
        
        # API 설정
        self.config_your_model = {
            "model": self.model,
            "max_tokens": 0.0,
            "temperature": self.config.temperature,
            "top_p": 0.0
        }
    
    def _make_prompt(self, text: str) -> List[Dict]:
        """프롬프트 생성"""
        # 템플릿의 깊은 복사
        messages = []
        for msg in self.template:
            messages.append({
                "role": msg["role"],
                "content": msg["content"].format(text=text) if "{text}" in msg["content"] else msg["content"]
            })
        return messages
    
    def _call_api_single(self, messages: List[Dict]) -> str:
        """단일 문장에 대한 API 호출"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": 1000  # 충분한 길이 확보
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            results = response.json()
            
            # 응답 내용 확인 및 로깅
            content = results["choices"][0]["message"]["content"]
            print(f"\n입력: {messages[-1]['content']}")
            print(f"출력: {content}")
            
            return content.strip()
            
        except Exception as e:
            print(f"API 호출 중 오류 발생: {str(e)}")
            return messages[-1]['content']  # 오류 시 원본 문장 반환

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터셋에 대한 실험 실행"""
        results = []
        for _, row in tqdm(data.iterrows(), total=len(data)):
            prompt = self._make_prompt(row['err_sentence'])
            corrected = self._call_api_single(prompt)
            results.append({
                'id': row['id'],
                'cor_sentence': corrected
            })
        return pd.DataFrame(results)

    def run_template_experiment(self, train_data: pd.DataFrame, valid_data: pd.DataFrame) -> Dict:
        """템플릿별 실험 실행"""
        print(f"\n=== {self.config.template_name} 템플릿 실험 ===")
        
        # 학습 데이터로 실험
        print("\n[학습 데이터 실험]")
        train_results = self.run(train_data)
        train_recall = evaluate_correction(train_data, train_results)
        
        # 검증 데이터로 실험
        print("\n[검증 데이터 실험]")
        valid_results = self.run(valid_data)
        valid_recall = evaluate_correction(valid_data, valid_results)
        
        return {
            'train_recall': train_recall,
            'valid_recall': valid_recall,
            'train_results': train_results,
            'valid_results': valid_results
        } 