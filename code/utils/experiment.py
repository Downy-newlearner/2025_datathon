import os
import time
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    
    def _make_prompt(self, text: str, examples: List[Dict]=None) -> List[Dict]:
        """프롬프트 생성"""
        # 템플릿의 깊은 복사
        messages = []
        for msg in self.template:
            # 템플릿의 {text}를 실제 입력 텍스트로 포맷팅
            if "{examples}" in msg["content"] and examples:
                # 예시를 문자열로 변환하여 포맷팅
                examples_content = "\n".join([ex["content"] for ex in examples])
                content = msg["content"].format(examples=examples_content)
            else:
                content = msg["content"].format(text=text) if "{text}" in msg["content"] else msg["content"]
            messages.append({
                "role": msg["role"],
                "content": content
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

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """두 문장 간의 코사인 유사도를 계산합니다."""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return similarity[0][0]

    def similarity_function(self, text: str) -> List[Dict]:
        """
        주어진 텍스트와 train.csv에서 샘플링한 문장들 간의 유사도를 계산하여,
        유사도가 높은 상위 m개의 문장을 프롬프트 템플릿의 예시 형태로 반환합니다.
        """
        m = 15
        n = 1000 # train.csv 에서 샘플링할 데이터 수

        # train.csv 파일 경로 지정
        train_csv = os.path.join(self.config.data_dir, 'train.csv')

        # train.csv 에서 1000개 데이터 샘플링
        train_data = pd.read_csv(train_csv)
        train_data = train_data.sample(n)

        # 샘플링한 데이터 중 문장 추출
        sentences = train_data['cor_sentence'].tolist()

        print(f"유사도 측정을 시작합니다. text: {text}")
        # sentences를 비교 text와 비교하여 유사도 측정
        similarities = []
        for sentence in sentences:
            similarity = self.calculate_similarity(text, sentence)
            similarities.append((sentence, similarity))

        # 유사도가 높은 20개 문장 반환
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 유사도가 높은 20개 문장을 프롬프트 템플릿의 예시 형태로 반환
        examples = []
        for sentence, similarity in similarities[:m]:
            examples.append({
                "role": "user",
                "content": f"예시 문장: {sentence}\n유사도: {similarity}"
            })
        return examples
    
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