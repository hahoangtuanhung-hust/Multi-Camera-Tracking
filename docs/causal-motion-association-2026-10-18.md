# Causal association không ReID — 2026-10-18

## Phạm vi

Theo chỉ đạo mới của người dùng: **không sử dụng ReID**, bao gồm không gọi
descriptor HSV trong ứng viên. Không tải checkpoint, không gửi crop/video ra
ngoài. Phần gallery vừa thử triển khai trong lượt này đã được gỡ; backend và
tests có từ trước vẫn được bảo toàn.

Default vẫn là `hybridsort`. Reference: `hybridsort-late-binding-balanced`.
Ứng viên thử nghiệm: `hybridsort-causal-association-v1`.
Detector, confidence 0.25, stride và commit delay 2 giây giữ nguyên reference.
Không bật lại terminal rejection, cascade IoU hay stationary reconnect đã fail.

## Bằng chứng và giả thuyết

Telemetry T4 cũ có 83 switch được phân loại local fragmentation, 66 long-gap,
37 short-gap, 55 cross-camera/global và 14 global-changed/local-stable.
Đây chỉ là dữ liệu chẩn đoán, không phải đối chứng cùng source cho code mới.

Kiểm tra code xác nhận:

- Nhánh thiếu appearance trước đây có thể cho retained track tham gia matcher
  thường. Nhánh causal lọc theo tuổi track bất kể descriptor có sẵn hay không.
- Confirmation recovery có thể tăng khi nhận cùng timestamp. Feature flag
  causal chặn việc đếm observation lặp.
- Recovery mơ hồ chỉ bị giới hạn bởi retention của track. Đã thêm deadline hold
  riêng và giới hạn candidate matrix; detection không được xét vẫn được trả về
  luồng unmatched, không bị âm thầm bỏ.

Với profile không appearance, các đường appearance recovery không được gọi.
Thử nghiệm đầu thay prediction ở vòng chính bằng observation velocity đã fail
smoke test và được tách sang `hybridsort-causal-extrapolation-rejected`.
Ứng viên hiện giữ nguyên hai vòng Hybrid-SORT và inertia; chỉ sau đó mới xét
motion rescue trên các cặp còn unmatched, yêu cầu ít nhất ba observation,
gap tối đa 0,8 giây, displacement chuẩn hóa tối đa một bbox, IoU dự đoán >=0,5,
size-log-change <=0,35, NIS <=9,21 và margin >=0,15. Không lấy detection đã có
owner từ vòng match trước. Đây là một ablation cần đo, không phải cam kết tăng
metric. Các guard cho appearance recovery cũng được kiểm thử độc lập, nhưng không
được coi là nguồn cải thiện nếu counter recovery bằng 0.

Chỉ sử dụng quá khứ của tracker; không nối track bằng toàn video, không dùng GT
trong association, không thay evaluator và không sửa các frame đã commit.

## Kiểm thử

245 unit/integration tests pass. Tests mới kiểm tra timestamp lặp, retained
track thiếu descriptor, hold deadline, motion veto, candidate budget không làm
mất detection, giữ commit delay và cấu hình không được gọi appearance extraction.

Smoke test phải dùng cùng cache 300 tick và video thật, lặp lại hai lần để
kiểm tra hashes. Chỉ khi không regression mới chạy replay 4.000 tick.
Metric 300 tick không đủ kết luận tăng IDF1 dài hạn.

Smoke test ứng viên motion-rescue đã hoàn tất hai lần cho cả reference và
candidate. Cả hai đạt IDF1 0,847419, HOTA 0,549253, precision 0,868708,
recall 0,858733, IDSW 6, fragmentation 13, public GID 26. Hash của prediction
và assignment (live lẫn committed) khớp giữa các lượt. Counter motion rescue
bằng 0: đây là kết quả không regression, **không phải bằng chứng cải thiện**.
Replay 300 tick bỏ đọc ảnh cũng khớp cả bốn hash, xác nhận đường chạy không
phụ thuộc pixel trong cấu hình này; số crop ReID bằng 0.

Source fingerprint đã kiểm thử:
`b1e7d7f80c3f6dc027b3992b2cb9f670c7c5d3c0f1e3f187a419cee704f5533a`.
Artifacts: `causal-rescue-reference-300.json`,
`causal-rescue-candidate-300.json`, `causal-rescue-noimage-300.json` trong
`data/cpu-audit`. Hai lượt 4.000 tick dùng cùng cache và source này; không dùng
thời gian chạy đồng thời trên CPU để suy ra hiệu năng T4.

## Controlled A/B 4.000 tick

Hai lượt hoàn tất trên cùng detection cache, cùng source fingerprint và không
đọc ảnh vì cả hai profile đều tắt appearance. Kết quả candidate và reference
giống hệt nhau, kể cả bốn hash prediction/assignment live và committed:

- IDF1 0,295004; HOTA 0,205235; AssA 0,123240.
- Precision 0,763252; recall 0,722513; IDSW 258; fragmentation 374.
- Public GID 166; median GID/GT 8; p95 12.
- Long-gap same-GID 0,118421; wrong-ID duration 33.590 frame.
- `causal_motion_rescues=0`; `reid_inference_crops=0`.

Artifacts: `causal-rescue-reference-4000.json` và
`causal-rescue-candidate-4000.json`. Prediction hash là
`967b4a3ba73d7b1d7e82d86033d592d8020536c8c11d93f35d3dcf63cdcd2906`;
assignment hash là
`4c920a1939c12e921681a7648f26a4dce4db6a4b26a2dd4cc5bf01bf922b3664`.

Kết luận: ablation motion rescue **không regression nhưng không tạo tác dụng**.
Nó nằm sau Hybrid-SORT hai vòng và có gate chặt hơn; các cặp đủ an toàn đã được
matcher nền nhận trước. Không promote candidate, không đổi default. Các guard
causal vẫn nằm sau feature flag để phục vụ kiểm thử/telemetry, nhưng không được
coi là cải thiện IDF1. Năm gate chất lượng chính chưa đạt: IDF1, HOTA, AssA,
IDSW và fragmentation; public GID, p95 GID/GT và long-gap rate cũng chưa đạt.

Thử nghiệm extrapolation đã bị loại: cùng source
`4d18915285d55b12627d4d672505f797b27eb9ec98e91f0cccf80be298328f0b`,
cùng cache/frame thật. Reference IDF1 0,847419; ứng viên 0,742828,
AssA 0,578951, recall 0,848991 và IDSW 20 thay vì 6. Artifacts:
`causal-motion-reference-300.json`, `causal-motion-candidate-300.json`.
Chỉ pass đầu của ứng viên hoàn tất; lượt repeat đã dừng vì regression rõ.

## Lệnh tái lập

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q

.\.venv\Scripts\python.exe benchmarks/replay_cpu.py `
  --detections data/cpu-audit/detections-low-floor-300.pkl `
  --out data/cpu-audit/causal-rescue-candidate-300.json `
  --only-current --profile hybridsort-causal-association-v1 `
  --video-frames --verify-determinism --skip-memory-profile --skip-stage-diagnostics
```

Reference dùng cùng lệnh, đổi profile thành `hybridsort-late-binding-balanced`
và output riêng. Không sửa source giữa hai lượt.

T4 A/B, nếu tiếp tục điều tra candidate experimental:

```powershell
$env:PYTHONUTF8 = '1'
uv run --extra modal modal run modal_app.py `
  --evaluate --max-frames 4000 --frame-offset 0 `
  --tracking-profile hybridsort-causal-association-v1 `
  --cameras camera_0001,camera_0007,camera_0008,camera_0010 `
  --metric-window-frames 500 --cumulative-frames 500,1000,2000,4000 `
  --metrics-out mot-metrics-causal-motion-4000-t4.json
```

Lệnh trên không phải bằng chứng đã chạy T4. Chưa có số CUDA/RSS hay latency
end-to-end mới. Không suy ra latency deployment từ replay detector cache.
Rollback bằng profile `hybridsort-late-binding-balanced`.

Render khuyến nghị dùng reference đã kiểm chứng; candidate chưa được promote:

```powershell
uv run --extra modal modal run modal_app.py `
  --seconds 60 --start 0 --stride 1 `
  --tracking-profile hybridsort-late-binding-balanced `
  --cameras camera_0001,camera_0007,camera_0008,camera_0010 `
  --out causal-motion-preview.mp4
```

Modal đóng gói source local khi chạy; `--upload` chỉ cần khi phải cập nhật
dataset/model trên volume. Không coi video render đẹp là bằng chứng giữ đúng ID.

## Giới hạn phạm vi

Chưa có calibration trên event slices/validation windows độc lập; các ngưỡng
rescue hiện là giả thuyết bảo thủ, không phải ngưỡng đã tối ưu theo GT.
Chưa bổ sung đầy đủ decision-event telemetry (capture/arrival/decision) riêng
cho rescue; hiện chỉ có counter tổng hợp. Không tuyên bố hoàn thành toàn bộ
thiết kế association trong yêu cầu ban đầu. Gallery/calibration neural ReID
được bỏ khỏi phạm vi theo yêu cầu mới. Motion-only không đảm bảo phục hồi đúng
identity sau long-gap khi nhiều người có vị trí/chuyển động tương tự.
