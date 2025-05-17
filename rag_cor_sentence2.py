import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from code.config import ExperimentConfig
from code.prompts.templates import TEMPLATES
from code.utils.experiment import ExperimentRunner

def main(num):

    UPSTAGE_API_KEYS = [
        "UPSTAGE_API_KEY1",
        "UPSTAGE_API_KEY2",
        "UPSTAGE_API_KEY3"
    ]


    # API 키 로드
    load_dotenv()
    api_key = os.getenv(UPSTAGE_API_KEYS[num])
    if not api_key:
        raise ValueError("API key not found in environment variables")
    
    # 기본 설정 생성
    base_config = ExperimentConfig(template_name='basic')
    
    # 데이터 로드
    test = pd.read_csv(os.path.join(base_config.data_dir, f'test{num+1}.csv')) # test.csv에서 100개의 데이터를 샘플링한 데이터입니다.
    
    # 템플릿 선택
    template_name = 'few_shot6_ver2'
    print(f"\n=== 테스트 데이터 예측 시작: 사용 템플릿: {template_name} ===")
    config = ExperimentConfig(
        template_name=template_name,
        temperature=0.0,
        batch_size=5,
        experiment_name="final_submission"
    )
    
    runner = ExperimentRunner(config, api_key)
    test_results = runner.run(test)
    
    output = pd.DataFrame({
        'id': test['id'],
        'cor_sentence': test_results['cor_sentence']
    })
    
    output.to_csv(f"1st_result_{template_name}_{num+1}.csv", index=False)
    print(f"\n제출 파일이 생성되었습니다: 1st_result_{template_name}_{num+1}.csv")
    print(f"사용된 템플릿: {template_name}")
    print(f"예측된 샘플 수: {len(output)}")

    # 유사도 비교 및 few-shot prompting
    # '1st_result_few_shot6_ver2.csv' 파일 로드
    result_df = pd.read_csv(f"1st_result_{template_name}_{num+1}.csv")
    results = []
    for _, row in tqdm(result_df.iterrows(), total=len(result_df)):
        text = row['cor_sentence']
        examples = runner.similarity_function(text)
        prompt = runner._make_prompt(text, examples)
        corrected = runner._call_api_single(prompt)
        results.append({'id': row['id'], 'cor_sentence': corrected})
        print(f"입력 문장: {text}")
        print(f"수정된 문장: {corrected}")

    # 결과를 CSV 파일로 저장
    output_df = pd.DataFrame(results)
    output_df.to_csv(f"submission_few_shot6_ver2_{num+1}.csv", index=False)
    print(f"\n결과가 CSV 파일로 저장되었습니다: submission_few_shot6_ver2_{num+1}.csv")


if __name__ == "__main__":
    main(1)
