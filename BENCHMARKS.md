## Quantization comparison (Measured after model has been loaded)

Test query: "What's the latest news on SRPT?" (5 retrieved articles, 644 prompt tokens)

| Quantization | File size | Load time | Generation speed | Peak memory |
|--------------|-----------|-----------|------------------|-------------|
| Q4_K_M       | 4.4 GB    | 9.427 s   | 10.8 tok/s       | 5016.9 MB   |
| Q8_0         | 7.7 GB    | 12.76 s   | 10.0 tok/s       | 1619.9 MB   |

Test query: "Why are analysts saying SRPT is bullish?" (5 retrieved articles, 642 tokens)

| Quantization | File size | Load time | Generation speed | Peak memory |
|--------------|-----------|-----------|------------------|-------------|
| Q4_K_M       | 4.4 GB    | 4.210 s   | 19.2 tok/s       | 4985.3 MB   |
| Q8_0         | 7.7 GB    | 2.102 s   | 9.94 tok/s       | 1657.4 MB   |

Test query: "Why are analysts saying SRPT is bearish?" (5 retrieved articles, 642 tokens)

| Quantization | File size | Load time | Generation speed | Peak memory |
|--------------|-----------|-----------|------------------|-------------|
| Q4_K_M       | 4.4 GB    | 2.385 s   | 21.8 tok/s       | 4935.5 MB   |
| Q8_0         | 7.7 GB    | 3.957 s   | 17.7 tok/s       | 1658.5 MB   |

Note: Q8_0 showing lower peak memory than Q4_K_M is counter to expectation
(the larger model would normally be expected to use more memory).
Not enough tests to determine the reason.

Generation speed favoring Q4_K_M across all three tests is
consistent with theory and is the more reliable finding here.
