## Quantization comparison (Measured after model has been loaded)

Test query: "What's the latest news on SRPT?" (5 retrieved articles, 644 prompt tokens)

| Quantization | File size | Load time | Prompt eval speed | Generation speed | Peak memory |
|--------------|-----------|-----------|---------------------|----------------|-------------|
| Q4_K_M       | 4.4 GB    | 3037 ms   | 212.2 tok/s         | 24.6 tok/s     | 4903.4 MB   |
| Q8_0         | 7.7 GB    | 2989 ms   | 215.7 tok/s         | 17.1 tok/s     | 4166.5 MB   |

Test query: "Why are analysts saying SRPT is bullish?" (5 retrieved articles, 642)

| Quantization | File size | Load time | Prompt eval speed | Generation speed | Peak memory |
|--------------|-----------|-----------|---------------------|----------------|-------------|
| Q4_K_M       | 4.4 GB    | 3002 ms   | 214.0 tok/s         | 26.4 tok/s     | 4726.6 MB   | 
| Q8_0         | 7.7 GB    | 2824 ms   | 227.4 tok/s         | 20.7 tok/s     | 2763.3 MB   |

Test query: "Why are analysts saying SRPT is bearish?" (5 retrieved articles, N/A)

| Quantization | File size | Load time | Prompt eval speed | Generation speed | Peak memory |
|--------------|-----------|-----------|---------------------|----------------|-------------|
| Q4_K_M       | 4.4 GB    | 3043 ms   | 211.1 tok/s         | 22.1 tok/s     | 5002.2 MB   | 
| Q8_0         | 7.7 GB    | 2748 ms   | 233.8 tok/s         | 19.2 tok/s     | 4680.9 MB   |