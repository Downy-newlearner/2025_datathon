import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from code.config import ExperimentConfig
from code.prompts.templates import TEMPLATES
from code.utils.experiment import ExperimentRunner

def main():
    # API 키 로드
    load_dotenv()
    api_key = os.getenv('UPSTAGE_API_KEY')
    if not api_key:
        raise ValueError("API key not found in environment variables")
    
    # 기본 설정 생성
    base_config = ExperimentConfig(template_name='basic')
    
    # 데이터 로드
    train = pd.read_csv(os.path.join(base_config.data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(base_config.data_dir, 'exp_test.csv')) # test.csv에서 100개의 데이터를 샘플링한 데이터입니다.
    
    # 토이 데이터셋 생성
    toy_data = train.sample(n=base_config.toy_size, random_state=base_config.random_seed).reset_index(drop=True)
    
    # train/valid 분할
    train_data, valid_data = train_test_split(
        toy_data,
        test_size=base_config.test_size,
        random_state=base_config.random_seed
    )
    
    # 모든 템플릿으로 실험
    results = {}
    for template_name in TEMPLATES.keys():
        config = ExperimentConfig(
            template_name=template_name,
            temperature=0.0,
            batch_size=5,
            experiment_name=f"toy_experiment_{template_name}"
        )
        runner = ExperimentRunner(config, api_key)
        results[template_name] = runner.run_template_experiment(train_data, valid_data)
    
    # 결과 비교
    print("\n=== 템플릿별 성능 비교 ===")
    for template_name, result in results.items():
        print(f"\n[{template_name} 템플릿]")
        print("Train Recall:", f"{result['train_recall']['recall']:.2f}%")
        print("Train Precision:", f"{result['train_recall']['precision']:.2f}%")
        print("\nValid Recall:", f"{result['valid_recall']['recall']:.2f}%")
        print("Valid Precision:", f"{result['valid_recall']['precision']:.2f}%")
    
    # 최고 성능 템플릿 찾기
    best_template = max(
        results.items(), 
        key=lambda x: x[1]['valid_recall']['recall']
    )[0]
    
    print(f"\n최고 성능 템플릿: {best_template}")
    print(f"Valid Recall: {results[best_template]['valid_recall']['recall']:.2f}%")
    print(f"Valid Precision: {results[best_template]['valid_recall']['precision']:.2f}%")
    
    # 최고 성능 템플릿으로 제출 파일 생성
    print("\n=== 테스트 데이터 예측 시작 ===")
    config = ExperimentConfig(
        template_name=best_template,
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
    
    output.to_csv("submission_baseline.csv", index=False)
    print("\n제출 파일이 생성되었습니다: submission_baseline.csv")
    print(f"사용된 템플릿: {best_template}")
    print(f"예측된 샘플 수: {len(output)}")

if __name__ == "__main__":
    main()