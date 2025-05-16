from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class ExperimentConfig:
    # 템플릿 설정
    template_name: str
    temperature: float = 0.0
    batch_size: int = 5
    experiment_name: Optional[str] = None
    
    # API 설정
    api_url: str = "https://api.upstage.ai/v1/chat/completions"
    model: str = "solar-pro"
    
    # 데이터 설정
    data_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    toy_size: int = 100
    random_seed: int = 42
    test_size: float = 0.2
    
    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"experiment_{self.template_name}"
        
        # 데이터 디렉토리 확인
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        # 필수 파일 존재 확인
        required_files = ['train.csv', 'test.csv']
        for file in required_files:
            file_path = os.path.join(self.data_dir, file)
            if not os.path.exists(file_path):
                raise ValueError(f"Required file not found: {file_path}") 