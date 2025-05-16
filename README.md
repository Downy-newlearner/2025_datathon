# 2025_datathon

## Directory 구조

- code/: Upstage에서 제공하는 베이스라인 코드입니다. main.py 파일만 루트 디렉토리로 옮겼습니다. 루트 디렉토리에 .env 파일을 생성하면 실행할 수 있습니다.
- data/: Upstage에서 제공하는 데이터들과 test.csv에서 100개의 데이터를 샘플링한 exp_test.csv이 존재하는 디렉토리입니다.
- main.py: 기존 main.py 코드에서 테스트 데이터셋 로드 부분을 수정했습니다. test.csv가 아닌 exp_test.csv를 로드합니다.

## API key 세팅하는 법

https://console.upstage.ai/api-keys
이 곳에서 Key를 가져와 .env 파일에 넣어주세요.

.env 파일은 프로젝트의 루트 디렉토리에 생성해야합니다.
