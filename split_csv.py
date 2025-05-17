import pandas as pd

# Read the original CSV file
data = pd.read_csv('/home/jdh251425/2025_datathon/data/test.csv')

# Calculate the number of rows for each split
num_rows = len(data)
chunk_size = num_rows // 3

# Split the data into three parts
chunk1 = data.iloc[:chunk_size]
chunk2 = data.iloc[chunk_size:2*chunk_size]
chunk3 = data.iloc[2*chunk_size:]

# Save each chunk to a separate CSV file
chunk1.to_csv('/home/jdh251425/2025_datathon/data/test1.csv', index=False)
chunk2.to_csv('/home/jdh251425/2025_datathon/data/test2.csv', index=False)
chunk3.to_csv('/home/jdh251425/2025_datathon/data/test3.csv', index=False)

print("CSV 파일이 세 부분으로 나뉘어 저장되었습니다.")
