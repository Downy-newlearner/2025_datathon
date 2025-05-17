import pandas as pd

# Read the original CSV file
data = pd.read_csv('/home/jdh251425/2025_datathon/data/test.csv')

# Calculate the number of rows for each split
num_rows = len(data)
chunk_size = num_rows // 9

# Split the data into nine parts
chunk1 = data.iloc[:chunk_size]
chunk2 = data.iloc[chunk_size:2*chunk_size]
chunk3 = data.iloc[2*chunk_size:3*chunk_size]
chunk4 = data.iloc[3*chunk_size:4*chunk_size]
chunk5 = data.iloc[4*chunk_size:5*chunk_size]
chunk6 = data.iloc[5*chunk_size:6*chunk_size]
chunk7 = data.iloc[6*chunk_size:7*chunk_size]
chunk8 = data.iloc[7*chunk_size:8*chunk_size]
chunk9 = data.iloc[8*chunk_size:]

# Save each chunk to a separate CSV file
chunk1.to_csv('/home/jdh251425/2025_datathon/data/test1.csv', index=False)
chunk2.to_csv('/home/jdh251425/2025_datathon/data/test2.csv', index=False)
chunk3.to_csv('/home/jdh251425/2025_datathon/data/test3.csv', index=False)
chunk4.to_csv('/home/jdh251425/2025_datathon/data/test4.csv', index=False)
chunk5.to_csv('/home/jdh251425/2025_datathon/data/test5.csv', index=False)
chunk6.to_csv('/home/jdh251425/2025_datathon/data/test6.csv', index=False)
chunk7.to_csv('/home/jdh251425/2025_datathon/data/test7.csv', index=False)
chunk8.to_csv('/home/jdh251425/2025_datathon/data/test8.csv', index=False)
chunk9.to_csv('/home/jdh251425/2025_datathon/data/test9.csv', index=False)

print("CSV 파일이 아홉 부분으로 나뉘어 저장되었습니다.")
