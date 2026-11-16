# Lệnh chạy CTIM / CTIM-G / CTIM-G-lazy với K = 5, 10, 15, 20

Cấu hình giống các log `*_lazy_h01.log` đã có: `--h 0.1 --paper-eval --items all`,
3 phương pháp `CTIM,CTIM-G,CTIM-G-lazy`. Chỉ thay đổi `--K`.

Log ghi ra thư mục gốc repo, tên `<dataset>_lazy_h01_K<K>.log`
(cả output lẫn lỗi/traceback đều nằm trong log).

## 0. Chuẩn bị (máy mới)

`data/` không nằm trong git — copy sang máy kia:

```
data/processed/digg/        data/processed/_calib_model.pkl
data/processed/ciao/        data/processed/_ciao.pkl
data/processed/delicious/   data/processed/_delicious.pkl
data/processed/epinions/    data/processed/_epinions.pkl
data/processed/lastfm/      data/processed/_lastfm.pkl
data/processed/yelp/        data/processed/_yelp_model.pkl
```

---

## 1. Năm dataset (digg, ciao, delicious, epinions, lastfm)

### Linux / macOS / Git Bash

```bash
cd marketing-im
for K in 5 10 15 20; do
  for pair in digg:_calib_model ciao:_ciao delicious:_delicious epinions:_epinions lastfm:_lastfm; do
    ds=${pair%%:*}; cache=${pair#*:}
    echo ">>> $ds  K=$K"
    python3 -u scripts/run_ctim_global.py \
      --dataset data/processed/$ds --cache data/processed/$cache.pkl \
      --K $K --h 0.1 --paper-eval --items all \
      --methods "CTIM,CTIM-G,CTIM-G-lazy" \
      > ${ds}_lazy_h01_K${K}.log 2>&1
  done
done
```

Xem tiến độ (mở terminal khác): `tail -f digg_lazy_h01_K5.log`

### Windows PowerShell

```powershell
cd marketing-im
$env:PYTHONIOENCODING = 'utf-8'
$caches = [ordered]@{ digg='_calib_model'; ciao='_ciao'; delicious='_delicious'; epinions='_epinions'; lastfm='_lastfm' }
foreach ($K in 5,10,15,20) {
  foreach ($ds in $caches.Keys) {
    Write-Host ">>> $ds  K=$K"
    cmd /c "python -u scripts/run_ctim_global.py --dataset data/processed/$ds --cache data/processed/$($caches[$ds]).pkl --K $K --h 0.1 --paper-eval --items all --methods CTIM,CTIM-G,CTIM-G-lazy > ${ds}_lazy_h01_K$K.log 2>&1"
  }
}
```

Xem tiến độ (mở cửa sổ khác): `Get-Content digg_lazy_h01_K5.log -Wait`

Thời gian tham khảo ở K=20: epinions ~15 phút, digg ~2 phút, ciao ~1 phút,
lastfm & delicious < 30 giây. K nhỏ hơn sẽ nhanh hơn.

---

## 2. Yelp

MC-IC và probe tắt (`--mc 0 --probe-sample 0`) như lần chạy Yelp trước — trên 366k
nút hai bước này quá tốn. Mỗi K mất khoảng 1 giờ trở lên → nên chạy trong `tmux`/`nohup`.

### Linux / macOS / Git Bash

```bash
cd marketing-im
for K in 5 10 15 20; do
  echo ">>> yelp  K=$K"
  python3 -u scripts/run_ctim_global.py \
    --dataset data/processed/yelp --cache data/processed/_yelp_model.pkl \
    --K $K --h 0.1 --paper-eval --items all \
    --methods "CTIM,CTIM-G,CTIM-G-lazy" --mc 0 --probe-sample 0 \
    > yelp_lazy_h01_K${K}.log 2>&1
done
```

Chạy nền, tắt terminal vẫn chạy tiếp:

```bash
cd marketing-im
nohup bash -c 'for K in 5 10 15 20; do python3 -u scripts/run_ctim_global.py --dataset data/processed/yelp --cache data/processed/_yelp_model.pkl --K $K --h 0.1 --paper-eval --items all --methods "CTIM,CTIM-G,CTIM-G-lazy" --mc 0 --probe-sample 0 > yelp_lazy_h01_K${K}.log 2>&1; done' > /dev/null 2>&1 &
```

### Windows PowerShell

```powershell
cd marketing-im
$env:PYTHONIOENCODING = 'utf-8'
foreach ($K in 5,10,15,20) {
  Write-Host ">>> yelp  K=$K"
  cmd /c "python -u scripts/run_ctim_global.py --dataset data/processed/yelp --cache data/processed/_yelp_model.pkl --K $K --h 0.1 --paper-eval --items all --methods CTIM,CTIM-G,CTIM-G-lazy --mc 0 --probe-sample 0 > yelp_lazy_h01_K$K.log 2>&1"
}
```

### Nếu máy kia chưa có dữ liệu Yelp

Chạy theo thứ tự. Bước cuối (Gibbs trên toàn bộ Yelp) rất lâu.

```bash
python3 scripts/stream_yelp.py --zip data/raw/yelp/Yelp-JSON.zip --out-dir data/raw/yelp/stream
python3 scripts/prepare_data.py yelp --raw-dir data/raw/yelp --out data/processed/yelp
python3 -u scripts/calibrate_h.py --dataset data/processed/yelp --cache data/processed/_yelp_model.pkl
```
