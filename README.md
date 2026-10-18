# Conveyor Bench — bộ benchmark cho "Offline AI Assistant for Conveyor Belt Manufacturing" (Arduino UNO Q)

Tài liệu này mô tả các benchmark trong thư mục, cách chạy, số liệu cần đọc và các giới hạn còn lại. Báo cáo kết quả gần nhất:
[BENCHMARK_REPORT.md](./BENCHMARK_REPORT.md).

> Trạng thái 2026-10-18: đã chạy benchmark SLM, STT/TTS, mô phỏng vòng kín, cascade và các kiểm tra vision trên PC Windows.
> Các số đo model/runtime là trên PC, **không phải trên Arduino UNO Q**. Bộ lệnh tiếng Việt hiện có chủ yếu tổng hợp; kiểm tra tiếng nói
> người thật mới dùng từ đơn tiếng Anh từ Speech Commands, không đại diện câu lệnh tiếng Việt trong xưởng.

---------------------------------------------------------------------------------------------------

## 1. Ý tưởng đánh giá (đọc phần này trước)

Kiến trúc của team: **SLM chỉ ĐỀ XUẤT → Safety & Invariant Arbiter QUYẾT ĐỊNH → STM32 THỰC THI**, MCU có bảo vệ độc lập
(quá dòng, quá nhiệt, mất encoder, kẹt, E-stop) và watchdog heartbeat từ Linux. Vì vậy đánh giá chia **hai tầng**:

| Tầng | Câu hỏi | Công cụ |
|---|---|---|
| **Từng khối** (STT, SLM, TTS, CV) | Mỗi khối đủ nhanh / đủ chính xác / đủ nhẹ RAM không? | `bench_*`, `workers_*` |
| **Hệ thống** (STT→SLM→Arbiter→MCU) | Hệ thống có **dừng khi cần** và **không bao giờ chạy lại khi không an toàn** không? | `arbiter.py`, cascade, `sim_closed_loop.py` |

Nguyên tắc chấm: nhãn do **luật xác định** sinh ra (không dùng LLM làm giám khảo); chỉ số an toàn báo kèm **cận trên Wilson 95 %**
(`hi95`) vì mẫu nhỏ không chứng minh được an toàn (0/25 lỗi vẫn có cận trên ≈ 13 %, 0/100 ≈ 3.7 %).

---------------------------------------------------------------------------------------------------

## 2. Cài đặt

```bash
pip install pandas numpy soundfile librosa faster-whisper llama-cpp-python onnxruntime opencv-python-headless \
            psutil matplotlib lm-eval transformers accelerate piper-tts      # tuỳ khối bạn chạy; không cần hết
```
* `arbiter.py`, `test_arbiter.py`, `conveyor_data.py`, `sim_closed_loop.py` **chỉ dùng thư viện chuẩn** → chạy được ngay, không cần model.
* Các worker mở rộng (`workers_*.py`) dùng lại `workers.py` trong workspace (định nghĩa `Meter`, `_load_refs`, `_norm`, `_wer`, `_ensure_vosk`
  và các kind gốc như `stt_fw`, `tts_piper`, `cls_ort`).
* Mô hình SLM: file `.gguf` (vd `model_cache/qwen2.5-0.5b-instruct-q4_k_m.gguf`).

---------------------------------------------------------------------------------------------------

## 3. Bản đồ file

### 3.1 Của bạn (giữ nguyên, mình chỉ chép vào để kiểm thử)
| File | Vai trò |
|---|---|
| `bench_common.py` | Hạ tầng: `bench()` (chạy mỗi model trong **tiến trình con**, cache JSON theo hash tham số), `table()`, `update_row()`, `plot_tradeoff()`, `prep_librispeech/imagenet/ocr`, diệt cây tiến trình khi timeout |
| `bench_ext.py` + `workers_ext.py` | STT/TTS/CV mở rộng: `stt_noise`, `stt_cmd`, `tts_quality`, `tts_judge`, `cv_latency`, `cls_corrupt`, `det_corrupt`, `cv_blockage`, `e2e` |
| `bench_lat.py` + `workers_lat.py` | `tts_correct`, `slm_latency`, `e2e_full`; bộ lệnh tiếng Anh `EN_VARIANTS`, `make_en_commands()` |
| `workers_task.py` | `slm_task`: benchmark tác vụ SLM (A/B/C) |
| `model_catalog.py` | Danh mục GGUF dùng chung cho benchmark tổng quát, latency và tác vụ; tải thủ công theo tên model |
| `conveyor_data.py` | Sinh dữ liệu + prompt + bộ chấm cho Task A/B/C (seed cố định, hash `b1a3d87fdf`); `--selftest` |
| `workers.py`, `record_commands.py`, các notebook `.ipynb` | Ở máy bạn (không có ở đây) |

### 3.2 Do mình viết
| File | Vai trò |
|---|---|
| `arbiter.py` | Arbiter tham chiếu, **fail-closed**, chỉ stdlib (xem §5) |
| `test_arbiter.py` | 5 nhóm test cho Arbiter (xem §5) |
| `workers_cascade.py` + `bench_cascade.py` | kind `cascade`: STT thật → SLM thật → Arbiter (xem §6) |
| `sim_closed_loop.py` | Mô phỏng vòng kín Plant/MCU/Linux + fault injection (xem §7) |
| `prep_real_data.py` | Chuẩn bị dữ liệu công khai: Speech Commands, tiếng máy DCASE/MIMII (xem §8) |
| `workers_real.py` + `bench_real.py` | kind `stop_kw`: lệnh DỪNG bằng giọng thật; `bench_real` là wrapper gom **mọi** kind |
| `eval_belt_video.py` | Chạy detector kẹt/dừng trên video băng tải thật bất kỳ (xem §8) |
| `cascade_core.py`, `run_cascade.py`, `record_cascade.py` | **Thiết kế cascade đời đầu** (3 đường keyword/slm/hybrid) + CLI ghi âm / chạy. `record_cascade.py` vẫn dùng được (xem §6.3). `cascade_core.py` hiện **không được worker nào gọi** (xem "Lưu ý" §6) |

**Chọn wrapper nào?** Mỗi wrapper bao gồm cái trước: `bench_common` ⊂ `bench_ext` ⊂ `bench_lat` ⊂ `bench_cascade` / `bench_real`.
Dùng **`from bench_real import *`** để gọi được tất cả kind (nó định tuyến kind → worker đúng).

---------------------------------------------------------------------------------------------------

## 4. Danh mục benchmark (tất cả kind đang có)

Gọi kiểu: `bx("thư_mục_kết_quả", "tên_model", "<kind>", {params})` → lưu `results_*/<thư_mục>/<tên>.json`; xem bảng bằng `table("thư_mục")`.

### 4.1 STT
| kind | Đo | Params chính |
|---|---|---|
| `stt_fw` (workers.py của bạn) | WER/RAM/latency cơ bản (LibriSpeech) | — |
| `stt_noise` | WER theo (loại ồn × SNR) | `data_dir`, `engine`/`size`, `kinds`, `snrs`, `noise_dir`, `n` |
| `stt_cmd` | Lệnh giọng nói: WER/CER, **intent_acc**, **false_restart**, **miss_stop**, latency/câu | như trên + `language`, `grammar`, `initial_prompt` |
| `stop_kw` (mới) | Lệnh **dừng** bằng **giọng người thật** (Speech Commands): miss_stop/false_stop (+hi95), recall từng từ | `data_dir` = thư mục do `prep_speech` tạo |

### 4.2 SLM
| kind | Đo | Params chính |
|---|---|---|
| (lm-eval trong notebook của bạn) | Độ chính xác chung + RAM GGUF | — |
| `slm_latency` | Thời gian load, TTFT, tok/s, tổng thời gian; theo độ dài ngữ cảnh, có/không prefix-cache; chịu tải nền, giới hạn lõi | `gguf`, `threads`, `cpus`, `hog_procs` |
| `slm_task` | **Task A** chọn tool (268 câu; negation, override, ASR-noise 0/1/2) · **Task B** SLM tự phán restart an toàn? (120 ca; **Dangerous Action Rate**) · **Task C** chẩn đoán vì sao dừng (105 ca; cause accuracy, biết nói UNKNOWN, **hallucination/grounding** bằng bộ kiểm claim xác định) | `gguf`, `tasks`, `limit`, `constrained`, `ablate` |

Danh sách SLM trong `edge_slm_benchmark.ipynb`, `latency_project.ipynb` và
`conveyor_task_eval.ipynb` lấy chung từ `model_catalog.py`. Catalog có Qwen3
(0.6B, 1.7B, 4B), Arch-Agent (1.5B, 3B), xLAM-2-1B, Gemma 3 1B,
Llama 3.2 1B, Phi-4 Mini, Phi-3.5 Mini, Gemma 4 E4B, Llama 4 Scout và
các baseline cũ.
Các notebook latency/tác vụ chỉ dùng GGUF đã có trong `model_cache`; để tải
model cụ thể, thêm tên trong `DOWNLOAD_MODELS` ở notebook. Riêng Llama 4
Scout là model 109B tổng (17B active), gồm nhiều shard khoảng 65 GB, cần
chấp nhận license Meta và bị bỏ qua khi chạy danh sách benchmark thông thường.
Các notebook latency/tác vụ cũng bỏ qua model `manual_only` trừ khi bật
`INCLUDE_MANUAL_MODELS` sau khi kiểm tra phần cứng và license.
Một số model cũng có license hạn chế hoặc yêu cầu chấp nhận điều khoản trước khi
tải; hãy kiểm tra `note` trong catalog trước khi dùng.

Các notebook tạo biểu đồ tập trung vào so sánh có ý nghĩa sau khi benchmark
chạy xong: `edge_slm_benchmark.ipynb` lưu `ram_vs_accuracy.png`,
`latency_project.ipynb` lưu `slm_latency_comparison.png` (TTFT và tốc độ sinh),
`conveyor_task_eval.ipynb` lưu `task_accuracy_by_model.png` và
`task_safety_rates_by_model.png` (tỷ lệ lỗi/an toàn, thấp hơn là tốt hơn).

### 4.3 TTS
| kind | Đo |
|---|---|
| `tts_quality` | TTFA, RTF, peak/clipping, im lặng đầu, âm lượng |
| `tts_judge` | Round-trip WER/CER (sạch + trong ồn) bằng STT cố định |
| `tts_correct` | TTS có đọc **đúng câu** không: exact-match, key-token recall (từ khoá/số), nhiều judge |

### 4.4 CV
| kind | Đo |
|---|---|
| `cv_latency` | Latency bền vững (pre+infer+post), p50/p95/p99, trôi nhiệt, chạy cùng tải nền |
| `cls_corrupt` / `det_corrupt` | Top-1 / mAP YOLO dưới tối/sáng/mờ/nhiễu/jpeg |
| `cv_blockage` | Phát hiện kẹt trên video băng tải **tổng hợp**: delay, tỉ lệ phát hiện, báo động giả, ms/khung |
| `eval_belt_video.py` | Cùng detector trên **video thật** → báo động giả/phút, trễ phát hiện dừng (không đo được "kẹt thật", xem §8) |
| `eval_mvtec_ad.py` | So sánh 3 baseline anomaly + MobileNetV3-Small/Large và EfficientNet-Lite0 trên MVTec AD |

MVTec AD được giải nén thành các thư mục category riêng (`bottle`, `cable`, ...),
mỗi category có `train/good`, `test/` và `ground_truth/`. Script tự tìm category
đặt trực tiếp trong `data/` (hoặc chỉ định thư mục khác bằng `--root`):

```bash
python eval_mvtec_ad.py                                      # mọi category, 3 CNN + 3 baseline
python eval_mvtec_ad.py --categories bottle metal_nut         # chọn category
python eval_mvtec_ad.py --classifiers mobilenet_v3_small efficientnet_lite0
```

Ngoài ba baseline truyền thống (`rgb_gaussian`, `rgb_spatial_knn`,
`texture_gaussian`), script dùng các ONNX ImageNet classifier sẵn có trong
`models/cv/`: MobileNetV3-Small (~9.7 MiB), MobileNetV3-Large (~20.9 MiB) và
EfficientNet-Lite0 (~17.7 MiB). Với mỗi CNN, vector logits 1000 lớp làm
descriptor ảnh; nearest-neighbor tới tối đa 32 ảnh `train/good` tạo anomaly
score. Đây là transfer baseline nhẹ, **không phải model anomaly chuyên biệt
đã train/fine-tune trên MVTec**. Nó chỉ cho image AUROC; pixel AUROC để trống
vì classifier global không định vị vùng lỗi. CSV ghi kích thước ONNX và thời
gian inference trung bình trên PC; thời gian không đại diện cho UNO Q.

YOLO là object detector (COCO), không phải anomaly detector. mAP COCO không
đo được lỗi sản phẩm trong MVTec, nên không dùng confidence YOLO như thể đó là
điểm defect. Hãy dùng benchmark YOLO/COCO riêng để so khả năng phát hiện vật
thể; kết quả MVTec này so khả năng biểu diễn/nhận ra ảnh bất thường.

### 4.5 Chuỗi & hệ thống
| kind / script | Đo |
|---|---|
| `e2e`, `e2e_full` | STT→NLU→(SLM)→TTS trong **một tiến trình**: RAM chung thật, thời gian tới **âm thanh đầu tiên** |
| `cascade` | STT thật → SLM chọn tool → **Arbiter**: tool_acc, miss_stop, false_restart, restart_success, spurious_stop, **unsafe_start (phải = 0)**, so với baseline từ khoá |
| `arbiter.py` + `test_arbiter.py` | Tính đúng/an toàn của Arbiter (không cần model) |
| `sim_closed_loop.py` | fault-to-stop, lớp nào dừng, chẩn đoán, restart, START cưỡng bức, soak dừng nhầm/giờ |

---------------------------------------------------------------------------------------------------

## 5. Arbiter (`arbiter.py`) và test

Hàm chính:
* `blockers(ctx, age_s)` — điều kiện **cho phép khởi động lại** (E-stop, cảm biến hợp lệ, dòng ≤ 2.5 A, nhiệt ≤ 70 °C, camera "belt empty", context không quá cũ).
* `running_blockers(ctx, age_s)` — điều kiện **phải dừng khi đang chạy** (camera "products moving" vẫn OK; "product stationary on belt" → BLOCKAGE; mất camera → CAMERA_UNAVAILABLE).
* `decide(tool, ctx, age_s)` → `request_stop` luôn STOP; `request_restart` chỉ START nếu không có blocker; mọi tool khác → NONE.
* `watchdog_action(ctx, age_s, running)` — dừng khi đang chạy mà vi phạm.
* **Fail-closed**: thiếu key, NaN/inf, sai kiểu, context cũ → coi là không an toàn.

Giả định chính sách **cần chốt với team** (chỉnh ở đầu file): `CURRENT_LIMIT_A=2.5`, `TEMP_LIMIT_C=70`, `MAX_CTX_AGE_S=1.0`, `STOP_ON_CAMERA_LOSS=True`.

```bash
python test_arbiter.py        # chạy ~1 giây, cần in PASS cho T1..T5
```
| Test | Nội dung |
|---|---|
| T1 | So sánh với oracle `conveyor_data.restart_violations` trên toàn lưới (2160 ca) |
| T2 | Bất biến tool × context × tuổi (168 486 ca): không bao giờ START khi có blocker; stop luôn STOP |
| T3 | Fuzz 20 000 context hỏng. Kèm **phát hiện**: oracle `restart_violations` **fail-open/ném exception ở 6931 ca** → đừng dùng nó làm Arbiter thật |
| T4 | Watchdog (8646 ca): dừng đúng khi vi phạm, **không dừng** khi chạy bình thường |
| T5 | Gold Task B (120 ca): Dangerous Action Rate = 0 |

---------------------------------------------------------------------------------------------------

## 6. Test chuỗi `cascade`

Đầu vào SLM là **chữ STT thật nhận ra** (khác Task A: nhiễu ASR giả lập bằng sửa chữ) nên lỗi STT lan sang quyết định cuối. STT + SLM chung **một tiến trình** → RAM chung thật.

### 6.1 Chạy
```bash
# (chỉ khi chưa có ghi âm thật) sinh bộ tổng hợp bằng Piper — số đo sẽ LẠC QUAN hơn người thật
python run_cascade.py --make-synthetic            # -> data/en_cascade_synth
python run_cascade.py --data data/en_cascade_synth --gguf model_cache/qwen2.5-0.5b-instruct-q4_k_m.gguf \
       --stt small.en --kinds machine --snrs 20 10 5
# noise thật: thêm  --kinds machine real --noise-dir data/noise
```
Hoặc từ notebook: `from bench_real import *` rồi `bx("cascade", "tên", "cascade", {...})` và `verdict_cascade(table("cascade"))`.

### 6.2 Chỉ số (cột `{slm|kw}_{chỉ số}_{điều kiện}`)
`tool_acc` · `miss_stop[_hi95]` · `false_restart[_hi95]` · `restart_success` · `spurious_stop` · **`unsafe_start`** (START khi trạng thái KHÔNG an toàn, 6 kiểu vi phạm × mọi câu — **phải = 0**) ·
`slm_valid_json` · `miss_stop_either` (bỏ sót stop ở **cả** SLM lẫn từ khoá — kiến trúc "dừng nếu một trong hai nghe ra stop") · `stt_ms_*`, `chain_ms_*` p50/p95 · `wer`.
`verdict_cascade()` (trong `bench_cascade.py`) cho cột PASS/FAIL — **ngưỡng tự đặt**, chỉnh theo yêu cầu nhóm và ghi lý do vào báo cáo.

Ý nghĩa: `unsafe_start = 0` chứng minh **Arbiter chặn được** khởi động nguy hiểm dù SLM sai. `false_restart > 0` nghĩa là Arbiter **không** cứu được lỗi *nghe nhầm ý định* khi máy đang an toàn — đây là rủi ro còn lại phải nêu trong báo cáo.

### 6.3 Ghi âm lệnh thật (khi có micro)
```bash
python record_cascade.py --speaker hieu --noise quiet     # ~72 clip; lặp lại với --noise fan/machine, và người nói khác
```
Ghi vào `data/en_cascade_real/refs.json` (nhãn `intent` = tên tool). Khuyến nghị ≥ 3 người × {yên tĩnh, ồn}.

> **Lưu ý (đã sửa trong lần rà này):** `record_cascade.py` ghi nhãn là *tên tool* (`request_stop`, `refuse_unsafe`...), còn `workers_cascade.py`
> trước đó chỉ chấp nhận nhãn ngắn (`stop`, `restart`...) và sẽ báo lỗi "intent lạ". Giờ chấp nhận cả hai. `run_cascade.py` cũng được sửa để đọc
> kết quả phẳng của worker (trước đó đọc khoá `rows` không tồn tại). `cascade_core.py` (đường *hybrid*, các chỉ số `unintended_start`...)
> là thiết kế đời đầu, **chưa nối vào worker** — chỉ `record_cascade.py` và `run_cascade.py --make-synthetic` còn dùng `pick_variants` của nó.
> Đã chạy cascade với faster-whisper `small.en` + Qwen2.5-0.5B trên câu lệnh tổng hợp; vẫn cần xác nhận bằng ghi âm operator thật.

---------------------------------------------------------------------------------------------------

## 7. Mô phỏng vòng kín (`sim_closed_loop.py`) — không cần board

```bash
python sim_closed_loop.py --all --hours 1     # in kết quả + ghi results_sim/{matrix,forced_start,soak}.csv
```
Mô hình 3 lớp (bước 10 ms): **Plant** (băng tải: vận tốc/dòng/nhiệt + lỗi chèn được) · **MCU** (an toàn thời gian thực độc lập Linux/AI: quá dòng, quá nhiệt, mất encoder/NaN, kẹt, E-stop, watchdog) · **Linux** (telemetry qua bridge, camera, `arbiter.py`, lệnh giọng nói).

Chạy 4 phần:
1. **Kịch bản vòng kín** (8 bước: chạy → kẹt → dừng → chẩn đoán → restart bị chặn → gỡ vật cản → restart OK ...).
2. **Fault injection**: JAM, OVERLOAD, OVERHEAT, ENC_DROP, NAN_CURRENT, ESTOP, CAM_DROP (+ mất liên lạc) × các biến thể Linux (`linux_ok`, `ai_hung` heartbeat gắn/không gắn với vòng an toàn, `linux_hung`, `bridge_down`) → fault-to-stop, **lớp nào dừng**, chẩn đoán đúng nguyên nhân (dùng lý do MCU chốt lại).
3. **START cưỡng bức** (giả lập Arbiter hỏng) → MCU tự chặn?
4. **Soak** không có lỗi thật → số **lần dừng nhầm/giờ** (kiểm tra debounce camera).

Kết quả đã có: kịch bản 8/8 PASS; ma trận PASS hết **trừ** `CAM_DROP` khi heartbeat **tách rời** vòng an toàn (đây là phát hiện thiết kế có chủ ý: nên gắn heartbeat vào vòng kiểm tra an toàn); START nguy hiểm qua Arbiter = 0; soak 1 h: debounce 0 ms → 10 dừng nhầm, 300 ms → 0.

> **Giới hạn:** mọi tham số vật lý và ngưỡng yêu cầu (`CFG`, `LIMIT_MS`, `TRUTH`) là **giả định**. Kết quả chứng minh **logic** an toàn, **không thay** số đo thật trên UNO Q.

---------------------------------------------------------------------------------------------------

## 8. Dữ liệu công khai (thay cho tự thu)

```bash
# Giọng người thật cho lệnh dừng (Google Speech Commands v0.02, CC BY 4.0)
python prep_real_data.py speech --src speech_commands_test_set_v0.02.tar.gz --out data/speech_cmds --n 100
python prep_real_data.py speech --download            # URL/dung lượng CHƯA xác minh; nếu lỗi hãy tải tay

# Tiếng máy thật làm nền ồn (DCASE 2020 Task 2 dev; CC BY-NC-SA 4.0 — ghi nguồn, không dùng thương mại)
python prep_real_data.py noise --src dev_data_ToyConveyor.zip --out data/noise --n 24
python prep_real_data.py noise --src dev_data_valve.zip --out data/noise --n 12 --append
```
Nguồn tải: Speech Commands test set v0.02; DCASE: https://zenodo.org/records/3678171 (ToyConveyor 1.9 GB, valve 1.0 GB). **Đừng tải MIMII gốc** (6.9–10.9 GB mỗi file).

Dùng:
```python
from bench_real import *
bx("stop", "small.en", "stop_kw", {"engine": "fw", "size": "small.en", "language": "en",
   "data_dir": "data/speech_cmds", "kinds": ["machine", "real"], "snrs": [20, 10, 5], "noise_dir": "data/noise"})
verdict_stop(table("stop"))            # đạt khi cận trên miss_stop ≤ 10 %, false_stop ≤ 5 % (ngưỡng tự đặt)
```
Video băng tải thật (vd bộ iron-ore của Mendeley — giấy phép chưa rõ, kiểm tra trước khi dùng):
```bash
python eval_belt_video.py --video belt.mp4 --fps 10 --size 160 120                 # báo động giả / phút
python eval_belt_video.py --video belt.mp4 --freeze-at 12                          # giả lập băng dừng giây 12 -> trễ phát hiện
```
**Giới hạn dữ liệu công khai:** clip Speech Commands dài 1 giây, **một từ** (không thay cho câu lệnh đầy đủ, không có câu phủ định/override); noise DCASE là mô hình máy đồ chơi/van, không phải xưởng của bạn; **không có nhãn "kẹt" công khai** → không đo được độ chính xác phát hiện kẹt thật, chỉ đo báo động giả và trễ dừng. Muốn có số thật cho các phần này cần dữ liệu tự tạo sau khi có thiết bị.

---------------------------------------------------------------------------------------------------

## 9. Quy trình gợi ý (theo thứ tự)

1. `python conveyor_data.py --selftest` và `python test_arbiter.py` — dữ liệu và Arbiter nhất quán.
2. `python sim_closed_loop.py --all` — logic an toàn vòng kín.
3. Từng khối: `stt_cmd`/`stt_noise`, `slm_latency` + `slm_task`, `tts_correct`/`tts_quality`, `cv_latency` + `cls_corrupt`/`det_corrupt`, `cv_blockage`.
4. Dữ liệu công khai: `prep_real_data.py` → `stop_kw`; `eval_belt_video.py`.
5. Hệ thống: `cascade` (và `e2e_full` cho RAM + latency chung).
6. **Trên board UNO Q** (khi có): chạy lại 3–5 để lấy RAM/latency thật; mọi số đo trên PC **không** đại diện cho QRB2210.

## 10. Còn thiếu (chưa làm được, cần board/thiết bị)

* Đo **fault-to-stop thật** (logic analyzer/GPIO) kể cả khi Linux bị `kill -9`/treo/OOM; **soak 4–8 giờ** với các container chạy đồng thời.
* Độ trễ và mất gói **bridge/RPMsg**; năng lượng; nhiệt độ dài hạn.
* Ghi âm lệnh đầy đủ (nhiều người, có ồn xưởng) và video băng tải có kẹt thật (có nhãn).
* Tiếng Việt (khi khách hàng yêu cầu): đã có sẵn `VI_*`/`prep_vi_commands` trong `bench_ext.py` và NLU tiếng Việt, nhưng `conveyor_data.py`/cascade hiện chỉ có tiếng Anh.
* Mọi ngưỡng PASS/FAIL (`verdict_*`, Arbiter, mô phỏng) là **giả định** — cần chốt với team/giảng viên rồi ghi nguồn trong báo cáo.
