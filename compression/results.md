# YoloV8
| Artifact | Size | Description |
| :--- | :--- | :--- |
| **FP32 .pt2** | 48.15 MB | Original full-precision model |
| **INT8 q/dq .pt2** | 59.37 MB | Intermediate ("bloated") |
| **XNNPACK .pte** | 10.91 MB | Deployable on Raspberry Pi |

---
+----------------------------------------------------------+
|                      FINAL RESULTS                       |
+----------------------------------------------------------+

  Model:                  weights/yolov8s.pt2
  Format:                 pt2
  Model size:             48.15 MB
  Total frames:           152

  Latency
    Mean:                   56.04 ms
    Min:                    49.39 ms
    Max:                    104.58 ms
    p50:                    54.15 ms
    p95:                    67.57 ms
    p99:                    79.88 ms

  Throughput
    Rolling FPS:            17.8
    Overall FPS:            17.8

+----------------------------------------------------------+

+----------------------------------------------------------+
|                      FINAL RESULTS                       |
+----------------------------------------------------------+

  Model:                  weights/yolov8s_int8_xnnpack.pte
  Format:                 pte
  Model size:             10.91 MB
  Total frames:           87

  Latency
    Mean:                   45.06 ms
    Min:                    43.18 ms
    Max:                    50.01 ms
    p50:                    44.75 ms
    p95:                    46.85 ms
    p99:                    50.01 ms

  Throughput
    Rolling FPS:            22.2
    Overall FPS:            22.2

+----------------------------------------------------------+