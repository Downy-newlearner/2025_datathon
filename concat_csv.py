import pandas as pd


def concat_csv(file_paths: list, output_file: str = 'submission.csv') -> None:
    """여러 개의 CSV 파일을 결합하여 하나의 CSV 파일로 저장합니다."""
    # 각 CSV 파일을 읽어오기
    dataframes = [pd.read_csv(file) for file in file_paths]

    # 데이터프레임을 하나로 합치기
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    # 합쳐진 데이터프레임을 저장
    concatenated_df.to_csv(output_file, index=False)
    print(f"CSV 파일이 성공적으로 결합되어 '{output_file}'로 저장되었습니다.")


if __name__ == "__main__":
    import sys
    # 파일 경로를 인자로 받기
    file_paths = [
        '/home/jdh251425/2025_datathon/submission_few_shot6_ver2_1.csv',
        '/home/jdh251425/2025_datathon/submission_few_shot6_ver2_2.csv',
        '/home/jdh251425/2025_datathon/submission_few_shot6_ver2_3.csv',
        '/home/jdh251425/2025_datathon/submission_few_shot6_ver2_4.csv',
        '/home/jdh251425/2025_datathon/submission_few_shot6_ver2_5.csv',
        '/home/jdh251425/2025_datathon/submission_few_shot6_ver2_6.csv',
        '/home/jdh251425/2025_datathon/submission_few_shot6_ver2_7.csv',
        '/home/jdh251425/2025_datathon/submission_few_shot6_ver2_8.csv',
        '/home/jdh251425/2025_datathon/submission_few_shot6_ver2_9.csv'
    ]
    concat_csv(file_paths)
