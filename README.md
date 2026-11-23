# mini-DATMO

用 LiDAR 點雲和已知位姿，自動產生移動物體的逐點標籤。

讀王傑智老師等人的 *Simultaneous Localization, Mapping and Moving Object Tracking*（IJRR 2007）時，我想試試 §7.1 的一致性判斷：如果一個位置之前是空的，現在卻出現了點，能不能用這個差異找出移動物體？

後來讀到 LMNet 和 AutoMOS，便參考 AutoMOS 的分群、追蹤和軌跡判斷流程，把幾何偵測的候選點整理成標籤，再拿 SemanticKITTI 的人工標籤評估。目前對移動較快的物體效果比較好，慢速和暫時停住的物體仍然容易漏掉。

## 目前的結果

先把標籤用在建圖上。下圖是序列 08 的 `3900:4071` 幀（右端不含），以 0.2 m voxel 累積的地圖。紅色是依人工標籤判定為移動、卻仍留在地圖裡的點，也就是「鬼影」。

![去除移動點前後的建圖比較](docs/figures/mapping_seq08_3900-4071.png)

不處理時有 13,101 個鬼影 voxel；用我的標籤去掉移動點後剩 3,787 個，用 LMNet 官方預測則剩 2,472 個。我的版本保留了 99.92% 的純靜態 voxel。這裡的鬼影數只計算完全由真值移動點組成的 voxel，定義和結果見 [mapping.py](mini_datmo/mapping.py) 與 [建圖結果](results/seq08_3900-4071.json)。

逐點分割的結果如下，數值單位都是 %。Precision、recall 和 IoU 是把評估範圍內各幀的 TP、FP、FN 加總後計算，實作在 [evaluate.py](mini_datmo/evaluate.py)。

| 方法 | 評估範圍 | Precision | Recall | IoU |
|---|---|---:|---:|---:|
| 我的版本 | 序列 08 全部 4,071 幀 | 67.9 | 42.3 | 35.3 |
| LMNet 官方預測 | 同上 | 75.5 | 72.1 | 58.4 |
| 我的版本 | 序列 07 全部 1,101 幀 | 54.1 | 43.9 | 32.0 |

來源：[序列 08](results/seq08_full_g5.json)、[序列 07](results/seq07_g5.json)，使用其中的 `all` 指標。LMNet 使用 LiDAR-MOS toy dataset 附的 SalsaNext 預測（1 張殘差影像），沒有另外訓練網路。LMNet 論文 Table I 報告的 IoU 是 59.9%；這裡用附帶預測重算是 58.4%，差異原因尚未確認。

另外，我用序列 08 的 `1500:1800` 和 `3000:3300` 共 600 幀調整方法。為了減少分群把相鄰物體黏在一起造成的誤判，最後只標記移動群中離候選點 0.5 m 以內的點（`refine=0.5`）。

| 方法 | Precision | Recall | IoU |
|---|---:|---:|---:|
| 我的版本 | 77.6 | 50.5 | 44.1 |
| 我的版本 + refine | 86.0 | 49.6 | 45.9 |
| LMNet 官方預測 | 69.3 | 71.6 | 54.4 |

這是開發時用來調整方法的範圍，不能當作完整序列的改善幅度。評估入口是 [checks/dev_eval.py](mini_datmo/checks/dev_eval.py)。

作為方法上的參考，AutoMOS Table I 在 SemanticKITTI 訓練序列上的結果是：

| 方法 | Precision | Recall | IoU |
|---|---:|---:|---:|
| Visibility-based baseline | 76.8 | 41.2 | 36.6 |
| AutoMOS 完整方法（ERASOR） | 88.1 | 82.5 | 74.2 |

我的版本和 visibility-based baseline 的 IoU 數字接近，但評估序列不同，還不能據此說兩者效果相當。目前也沒有重現 AutoMOS 的完整方法。

## 方法

先用資料集的位姿對齊點雲，去掉地面和車身附近的點，再比較不同時間的可見性：把目前的點轉到參考幀的感測器座標，如果它落在參考幀光束曾經穿過的空間，就列為候選移動點。這部分從 IJRR 2007 §7.1 的一致性判斷出發，實作在 [polar.py](mini_datmo/polar.py) 和 [consistency.py](mini_datmo/consistency.py)。目前的基準設定比較前、後各 5 幀，可以使用未來資料，因為目標是離線標註。

接著用 DBSCAN 分群，以等速 Kalman filter 和匈牙利演算法追蹤。最後檢查整條 track 的淨位移、直線度和候選點比例，通過的 track 才標成移動。這部分參考 AutoMOS 的流程，但使用的分群、配對和軌跡條件並不完全相同。實作在 [cluster.py](mini_datmo/cluster.py)、[tracker.py](mini_datmo/tracker.py) 和 [labeler.py](mini_datmo/labeler.py)。

## 做的時候遇到什麼

一開始沒有排除自己車身的點。這些點在感測器座標裡幾乎不動，轉到世界座標後卻跟著車子走，結果被追蹤成一條很長的「移動物體」軌跡。後來在前處理排除了車身附近的點，才解決這個誤判。

另一個問題是停住的車。序列 08 開頭有兩台車停在路口，觀察期間幾乎沒有移動，但人工標籤仍標成移動。這讓我想知道，整體 recall 偏低到底是所有物體都抓不好，還是主要漏在慢速和停住的物體，所以另外做了速度分組：

| 估計速度 | 占真值移動點（約） | 我的 recall | LMNet recall |
|---|---:|---:|---:|
| < 0.5 m/s | 21% | 0.0% | 42.1% |
| 0.5–2 m/s | 28% | 15.8% | 62.5% |
| > 2 m/s | 51% | 74.2% | 89.9% |

這是序列 08 完整序列、未加 refine 的結果，來源是 [速度分組結果](results/seq08_g5_by_speed.json)。速度由真值物體點雲中心在前後幀的位移估計；遮擋和可見表面改變也會讓中心漂移，因此分組只是近似。

觀察：差距在慢速和停住的組別特別明顯。LMNet 對停住的物體也有部分召回，我猜可能和訓練學到的特徵有關，但這個實驗還不能確認原因。

我也試過拉長比較的時間範圍，希望找出短時間內看不出來的移動。候選點的 recall 確實提高了，但完整序列的最終 IoU 反而下降：多間隔比較是 30.6%，長時間比較是 31.3%，都低於基準的 35.3%（[多間隔結果](results/seq08_full_g5-10-20.json)、[長時間結果](results/seq08_full_h145v3.json)）。

接下來最想弄懂的是這個差距：AutoMOS 用 ERASOR 產生候選，再透過後續流程把 precision 拉高；我的版本增加時間範圍後，卻沒能把多出來的誤判有效濾掉。我想先查清楚問題出在候選、分群還是追蹤，再決定要改哪一段。

## 執行

在 repo 根目錄安裝依賴：

```bash
pip install -r mini_datmo/requirements.txt
```

序列 08 使用 [LiDAR-MOS](https://github.com/PRBonn/LiDAR-MOS) 提供的 toy dataset，包含點雲、位姿、人工標籤和 LMNet 預測，解壓到 `datasets/lmnet_toy_seq08/`。其他 SemanticKITTI 序列可用下載腳本透過 HTTP Range 取得指定序列的點雲：

```bash
python mini_datmo/tools/fetch_sequences.py 07 --out datasets/semantickitti
```

人工標籤和位姿另從 [SemanticKITTI 標籤包](https://www.semantic-kitti.org/assets/data_odometry_labels.zip) 取得。序列 07 的資料目錄是 `datasets/semantickitti/dataset/sequences/07/`。

```bash
cd mini_datmo

# 基準版本：輸出逐點標籤和評估結果
python run.py --data seq08 --out out/seq08_g5 --json out/results/seq08_full_g5.json

# 依速度分組評估，以及建圖比較
python eval_by_speed.py --pred out/seq08_g5/predictions
python mapping.py --pred out/seq08_g5/predictions --frames 3900:4071

# 在開發範圍比較 refine 前後
python checks/dev_eval.py
python checks/dev_eval.py "refine=0.5"

# 另一個序列
python run.py --data seq07 --out out/seq07_g5
```

## 參考

- C.-C. Wang, C. Thorpe, M. Hebert, S. Thrun, H. Durrant-Whyte. *Simultaneous Localization, Mapping and Moving Object Tracking*. IJRR, 2007.
- X. Chen et al. *Moving Object Segmentation in 3D LiDAR Data: A Learning-Based Approach Exploiting Sequential Data*. RA-L, 2021.
- X. Chen et al. *Automatic Labeling to Generate Training Data for Online LiDAR-Based Moving Object Segmentation*. RA-L, 2022.
- J. Behley et al. *SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences*. ICCV, 2019.
