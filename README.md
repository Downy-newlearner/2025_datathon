## 핵심 플로우
일반 문장 교정을 요청합니다
1차 교정된 문장과 train.csv 데이터를 유사도 계산하여 유의미한 예시를 선정합니다.
선정된 예시가 포함된 프롬프트를 통해 2차 문장 교정을 진행한다.

## Retrieval-Augmented Generation(RAG) 적용
입력 문장과 유사한 오류 유형을 가진 예시를 프롬프트에 삽입하기 위해 RAG(Retrieval-Augmented Generation) 전략을 적용함

임베딩 방식:
- 일반적인 딥러닝 기반 모델(BERT 등) 대신, TfidfVectorizer (sklearn) 를 활용하여 문장을 벡터화함
- 계산 속도가 빠르고 학습 없이도 문장 간 유사도를 계산할 수 있어 Solar 환경에서도 가볍게 연동 가능

검색 방식:
- 전체 train 데이터에서 예시 문장들을 미리 TF-IDF 벡터로 변환
- 입력 문장을 동일하게 TF-IDF 벡터화한 후, 코사인 유사도로 top-k 유사 예시 검색

검색된 예시는 {examples} 자리에 삽입되어 few_shot6_ver3 프롬프트에 포함됨


## 특징
✅ 1. RAG 기반 동적 예시 삽입으로 Recall 성능 향상
기존 고정형 few-shot 프롬프트(few_shot6_ver2)는 다양한 문장에 대응하는 데 한계가 있었음.

이를 보완하기 위해 입력 문장과 유사한 오류 유형을 포함한 예시를 검색하여 프롬프트에 삽입하는 Retrieval-Augmented Generation(RAG)을 적용.

결과적으로 복합 오류 문장(띄어쓰기 + 조사 오류 등)에서 일반화 능력이 향상됨.

✅ 2. 프롬프트 구조화로 실험 확장성과 재현성 확보
교정 예시와 입력 문장을 분리하여 삽입할 수 있도록 구조화함으로써, RAG와의 통합이 용이해졌고, 다양한 샷 수 및 예시 조합 실험이 프롬프트 코드 수정 없이 가능해졌음.

## 결과 요약
🏆 리더보드 점수
Recall : 75.0991, Precision   : 75.0594
-> 6th 


