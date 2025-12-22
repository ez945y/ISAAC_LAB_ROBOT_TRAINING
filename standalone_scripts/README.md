# Single Scripts - Isaac Lab 實驗腳本

這個目錄包含獨立的 Isaac Lab 實驗腳本，用於快速原型開發和學習。

## 腳本說明

### 01_basic_auto_drive.py - 基礎場景 + 自動駕駛
基本的 Isaac Lab 場景設置，包含：
- Jetbot 移動機器人
- 可變形立方體 (Deformable Cube)
- 靜態碰撞支架
- 自動駕駛邏輯（直行和轉彎）

```bash
python 01_basic_auto_drive.py --num_envs 1
```

### 02_keyboard_control.py - 鍵盤控制
在基礎場景上增加鍵盤控制功能：
- WASD / 方向鍵控制 Jetbot
- 命令平滑化處理（防止抖動）
- 差速驅動模型 (Differential Drive)
- 按 R 重置場景

```bash
python 02_keyboard_control.py --num_envs 1
```

### 03_domino_fpv.py - 骨牌 + 第一人稱視角
包含更複雜的物理互動：
- 4 個高度遞增的骨牌，展示物理模擬效果
- 橘色可變形立方體懸掛在空中（頂部節點固定）
- 第一人稱視角鏡頭跟隨 Jetbot
- 鍵盤控制 + 場景重置

```bash
python 03_domino_fpv.py --num_envs 1
```

### 04_trajectory_record.py - 軌跡記錄/播放 ⭐
最完整的實驗腳本，包含所有功能：

#### 功能特色
- **鍵盤控制**: WASD 或方向鍵控制 Jetbot
- **骨牌物理**: 4 個高度遞增的骨牌
- **可變形物體**: 橘色立方體懸掛在空中
- **軌跡記錄/播放**: 
  - 按 `I` 開始記錄軌跡
  - 按 `O` 停止記錄並開始循環播放
  - 按 `R` 重置場景並清除軌跡
- **第一人稱視角**: 鏡頭自動跟隨 Jetbot

#### 控制方式
| 按鍵 | 功能 |
|------|------|
| W / ↑ | 前進 |
| S / ↓ | 後退 |
| A / ← | 左轉 |
| D / → | 右轉 |
| R | 重置場景 |
| I | 開始記錄軌跡 |
| O | 停止記錄，開始播放 |

```bash
python 04_trajectory_record.py --num_envs 1
```

## 腳本演進關係

```
01_basic_auto_drive.py     (基礎)
        ↓
02_keyboard_control.py     (+鍵盤控制)
        ↓
03_domino_fpv.py           (+骨牌 +第一人稱視角)
        ↓
04_trajectory_record.py    (+軌跡記錄/播放)
```

## 注意事項

1. 這些腳本是獨立的，不需要安裝任何 extension
2. 確保 Isaac Sim 視窗有焦點才能接收鍵盤輸入
3. 可以使用 `--device` 參數指定 CPU 或 CUDA 設備

## 依賴

- Isaac Lab 2.2+
- NVIDIA Isaac Sim 5.0+
