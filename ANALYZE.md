# Gemmini Tiled Matmul Auto: E2E 동작 분석 (WS 모드)

> 실행 모드: **Weight Stationary (WS)**  
> 예시: A(45×768), B(768×2304), C(45×2304)  
> 하드웨어: DIM=16, Scratchpad=16,384 rows, Accumulator=1,024 rows

---

## 1. 전체 흐름 요약

```
CPU                         NPU (Gemmini)
─────────────────────────────────────────────────────────
A,B,C를 DRAM에 배치
     ↓
tiled_matmul_auto 호출
     ↓
[Auto Tiling] ───────→  tile_I=3, tile_J=7, tile_K=48 결정
     ↓
[Outer Loop 21회]
  ├─ j0=0:  C[:,   0: 111]
  ├─ j0=1:  C[:, 112: 223]
  └─ ... (총 21개 타일)

각 타일 낮부:
  [MVIN A,B]  ──→  DRAM → Scratchpad
  [WS Loop]   ──→  Hardware Loop (K→J→I)
    ├─ [Preload B] →  Scratchpad → PE 레지스터
    ├─ [Compute]   →  A 스트리밍, PE에서 A×B 누적
    └─ [MVOUT C]   →  Accumulator → DRAM
     ↓
[Fence] 완료 대기 → CPU가 C 사용
```

---

## 2. Scratchpad & Accumulator 용량 계산

### 2.1 하드웨어 용량 공식

Gemmini 메모리는 **row-addressed** 방식입니다.  
여기서 **1 row** = **DIM(16)개의 원소**를 담는 메모리 단위입니다.

| 구성요소 | 파라미터 | 총 row 수 | 1 row 크기 | 총 용량 계산 |
|----------|----------|-----------|------------|--------------|
| **Scratchpad** | BANK_NUM(4) × BANK_ROWS(4096) | **16,384 rows** | 16개 int8 = **16 bytes** | 16,384 × 16 B = **262,144 B = 256 KB** |
| **Accumulator** | ACC_ROWS(1024) | **1,024 rows** | 16개 int32 = **64 bytes** | 1,024 × 64 B = **65,536 B = 64 KB** |

> **왜 row 단위인가?**  
> Systolic Array는 DIM(16)×DIM(16) 크기로 데이터를 처리합니다. 메모리 접근도 이 단위로 이루어져서, 1 row가 정확히 16개 원소를 담습니다.

### 2.2 타일이 차지하는 용량

타일 하나가 차지하는 **row 수**와 **bytes**:

```
[Scratchpad]
A 타일: tile_I × tile_K = 3 × 48 = 144개의 16×16 블록
       = 144 × 16 rows = 2,304 rows
       = 2,304 × 16 B = 36,864 B

B 타일: tile_K × tile_J = 48 × 7 = 336개의 16×16 블록
       = 336 × 16 rows = 5,376 rows
       = 5,376 × 16 B = 86,016 B

[Accumulator]
C 타일: tile_I × tile_J = 3 × 7 = 21개의 16×16 블록
       = 21 × 16 rows = 336 rows
       = 336 × 64 B = 21,504 B

[총합]
Scratchpad: A + B = 36,864 + 86,016 = 122,880 B
Accumulator: C = 21,504 B
```

### 2.3 WS 모드 Double-Buffer 제약

WS 모드는 **절반 용량만** 사용 (나머지 절반은 다음 타일 prefetch용):

| 구성요소 | 총 용량 | 사용 가능 | 타일 사용량 | 비율 |
|----------|---------|-----------|-------------|------|
| **Scratchpad** | 256 KB | **128 KB** | 122,880 B (120 KB) | 93.8% |
| **Accumulator** | 64 KB | **32 KB** | 21,504 B (21 KB) | 65.6% |

```
검증:
  Scratchpad: 122,880 B ≤ 131,072 B (128 KB) ✓
  Accumulator: 21,504 B ≤ 32,768 B (32 KB) ✓
```

### 2.4 타일 크기가 용량을 초과하면?

```
만약 tile_J=8이면:
  A = 3×48×16 rows = 2,304 rows = 36,864 B
  B = 48×8×16 rows = 6,144 rows = 98,304 B
  총 = 8,448 rows = 135,168 B > 131,072 B (128 KB) ✗
  
→ Scratchpad 한도 초과! tile_J=7까지만 가능
```

---

## 3. Auto Tiling: 용량에 맞춘 타일 크기 결정

### 3.1 하드웨어 용량

| 구성요소 | 총 용량 | WS 모드 사용 가능 | 의미 |
|----------|---------|-------------------|------|
| **Scratchpad** | 16,384 rows | **8,192 rows** | 절반은 다음 타일 prefetch용 |
| **Accumulator** | 1,024 rows | **512 rows** | 절반은 다음 타일 prefetch용 |

> WS 모드는 **Double Buffering**: 현재 타일을 Compute하는 동안, 다음 타일을 미리 적재

### 3.2 타일 크기 계산

입력을 DIM(16)의 배수로 패딩:
- I: 45 → **48** (패딩 3)
- K: 768 → **768** (패딩 0)
- J: 2304 → **2304** (패딩 0)

용량 제약 내 Greedy Expansion:
```
1. tile_I=3, tile_J=5, tile_K=48 → 6,144 rows ✓
2. tile_J=7까지 확장 → 7,680 rows ✓ (8,192 한도 내)
3. tile_J=8 → 8,448 rows ✗ (한도 초과)
```

**최종 타일 크기**:
| 타일 | 값 | 실제 크기 |
|------|-----|-----------|
| tile_I | **3** | 3×16 = **48행** |
| tile_J | **7** | 7×16 = **112열** |
| tile_K | **48** | 48×16 = **768열** |

### 3.3 전체 행렬 분할

```
C 행렬: 48행 × 2304열

┌──────────┬──────────┬──────────┬───────┬──────────┐
│ 112열    │ 112열    │ 112열    │  ...  │ 64열     │
│ j0=0     │ j0=1     │ j0=2     │       │ j0=20    │
└──────────┴──────────┴──────────┴───────┴──────────┘

총 21개 타일 (J0=21)
각 타일: K=768 전체와 곱해짐 (K0=1)
```

---

## 4. E2E 단계별 흐름

### 4.1 단계 0: CPU가 행렬을 DRAM에 배치

```
CPU DRAM
├─ A: 45×768 (int8) → 실제 48×768 (패딩 3행)
├─ B: 768×2304 (int8)
├─ C: 45×2304 (int8) ← 출력용
└─ D: bias (없음)
```

### 4.2 단계 1: NPU 설정 (Config)

CPU가 RoCC 명령으로 NPU 파이프라인 설정:

```
[Config_EX]  실행 모드
  ├── Dataflow: WS (Weight Stationary)
  └── Activation: NONE

[Config_LD]  Load 설정
  ├── A: stride=768, scale=1.0
  ├── B: stride=2304, scale=1.0
  └── D: 없음

[Config_ST]  Store 설정
  └── stride=2304, scale=1.0, activation=NONE
```

### 4.3 단계 2: 타일 적재 (MVIN)

**한 타일(j0=0)의 MVIN:**

```
MVIN A (DRAM → Scratchpad)
├─ 크기: 48행 × 768열
├─ A 주소: DRAM → Scratchpad row 0 (Bank Set 0)
└─ 단위: 16×16 블록 (DIM×DIM)

MVIN B (DRAM → Scratchpad)
├─ 크기: 768행 × 112열
├─ B 주소: DRAM → Scratchpad row 2816 (끝에서 역순)
└─ 단위: 16×16 블록
```

**데이터 변환:**
```
DRAM (int8, row-major) ──[MVIN]──→ Scratchpad (int8, row-addressed)
```

### 4.4 단계 3: WS 하드웨어 루프

**loop_ws 설정 (5개 명령):**
```
Bounds:     I=3, J=7, K=48
Addresses:  A_dram, B_dram, C_dram
Strides:    A=768, B=2304, C=2304
Loop:       a_spad_id=1 (Bank Set 0 사용)
```

**루프 실행 (K→J→I 순서):**

```
K=0..47:
  J=0..6:
    I=0..2:
      
      ┌────────────────────────────────────────┐
      │ j==0일 때: MVIN A (DRAM→Scratchpad)    │
      │ i==0일 때: MVIN B (DRAM→Scratchpad)    │
      │                                         │
      │ i==0일 때: PRELOAD B (Scratchpad→PE)   │
      │            → PE[0:15][0:15]에 B 고정    │
      │                                         │
      │ COMPUTE: A 스트리밍 입력                │
      │          PE에서 A×B 누적               │
      │          → Accumulator에 저장          │
      │                                         │
      │ k==47일 때: MVOUT C (Accumulator→DRAM) │
      │            → scale + activation        │
      └────────────────────────────────────────┘
```

**총 반복: 48×7×3 = 1,008회**

### 4.5 단계 4: Systolic Array 동작

**한 Compute의 낮부:**

```
PE는 16×16 그리드

B는 PE 레지스터(pe_state)에 고정:
  PE[0][0] = B[0][0], PE[0][1] = B[0][1], ..., PE[15][15] = B[15][15]

사이클별 동작:
  사이클 0: A[0][0] 입력 → PE[0][0]가 A[0][0] × B[0][0] 누적
  사이클 1: A[0][1] 입력 → PE[0][1]가 A[0][1] × B[0][1] 누적
  ...
  사이클 15: A[0][15] 입력 → PE[0][15]가 A[0][15] × B[0][15] 누적

→ result[0][0:15] = Σ(k=0..15) A[0][k] × B[k][0:15]
```

**WS 모드 핵심:**
```
B는 PE에 고정 → Scratchpad에서 B를 읽지 않음
A만 매 사이클 새로 입력 → 메모리 대역폭 절약
```

**데이터 타입 변화:**
```
Scratchpad (int8) ──→ PE 입력 (int8)
                          ↓
                    곱셈 (int8 × int8)
                          ↓
                    누적 (int32)
                          ↓
                    Accumulator (int32)
```

### 4.6 단계 5: 결과 저장 (MVOUT)

**k==K-1(47)일 때만 실행:**

```
Accumulator (int32) ──[MVOUT]──→ DRAM (int8)
         ↓
    [Scale]     float 곱셈 + 반올림
         ↓
    [Activation]  ReLU 등 (있는 경우)
         ↓
    [Clipping]    -128 ~ 127
         ↓
    [DRAM]        int8로 저장
```

### 4.7 단계 6: 다음 타일 (Double Buffer 전환)

```
j0=0: Bank Set 0 사용
  ├── Compute 중...
  └── 동시에: j0=1의 A를 Bank Set 1에 MVIN (prefetch)

j0=1: Bank Set 1 사용
  ├── Compute 중...
  └── 동시에: j0=2의 A를 Bank Set 0에 MVIN (prefetch)
```

**효과:** Compute가 끝나면 다음 타일이 이미 적재됨 → **Compute와 DMA overlap**

---

## 5. 메모리와 연산 단위 변화

### 5.1 한 타일에서의 단위 변화

| 단계 | 메모리 | 단위 | 타입 | 크기 |
|------|--------|------|------|------|
| A (DRAM) | CPU | 48×768 | int8 | 36,864 B |
| B (DRAM) | CPU | 768×112 | int8 | 86,016 B |
| MVIN A,B | DRAM→Scratchpad | 16×16 블록 | int8 | 256 B |
| Preload B | Scratchpad→PE | 16×16 | int8 | 256 B |
| Compute | PE 낮부 | 16×16×16 MAC | int8×int8→int32 | 4,096 ops |
| Accumulator | PE→Acc | 48×112 | int32 | 21,504 B |
| MVOUT C | Acc→DRAM | 16×16 블록 | int32→int8 | 256 B |

### 5.2 연산량

```
한 타일 (48×112 결과):
  곱셈-누적 = 48 × 112 × 768 = 4,128,768회

전체 행렬 (45×2304):
  총 타일 = 21개
  총 연산 = 45 × 2304 × 768 = 79,626,240회
```

---

## 6. CPU vs NPU 동작 분리

### 6.1 CPU 역할

| 단계 | 동작 |
|------|------|
| **준비** | A, B, C 행렬 DRAM에 할당/초기화 |
| **설정** | RoCC 명령으로 NPU 설정 (Config, Loop_WS) |
| **대기** | gemmini_fence()로 NPU 완료 대기 |
| **사용** | C 행렬 읽기 |

### 6.2 NPU 역할

| 컨트롤러 | 동작 |
|----------|------|
| **Load** | MVIN: DRAM → Scratchpad (DMA) |
| **Execute** | Preload (B→PE), Compute (A×B 누적) |
| **Store** | MVOUT: Accumulator → DRAM (scale+activation) |

### 6.3 동시 실행

```
시간 →
─────────────────────────────────────────────────
CPU:  [설정]..................[Fence]
NPU:       [MVIN][Compute][MVOUT]
               ↓
          다음 타일 MVIN (동시에)
               ↓
          Double Buffer 전환
```

---

## 7. WS 모드 핵심

### 7.1 왜 WS 모드인가?

| 항목 | OS 모드 | WS 모드 |
|------|---------|---------|
| B 접근 | 매번 Scratchpad 읽음 | PE 레지스터 사용 |
| 타일 크기 | tile_J=18 (큼) | tile_J=7 (작음) |
| Compute | 2,592회 | 1,008회 |
| B 재사용 | 없음 | **I=3번 재사용** |
| Double Buffer | 없음 | **있음** |

**결론:** K가 클 때 B 재사용 효과가 크고, 메모리 대역폭을 절약할 수 있음

### 7.2 데이터 흐름 요약

```
┌─────────┐   MVIN    ┌───────────┐   Preload   ┌──────────┐
│  DRAM   │ ────────→ │ Scratchpad│ ──────────→ │    PE    │
│ (A,B)   │           │  (int8)   │   (B만)     │ (16×16)  │
└─────────┘           └───────────┘             └────┬─────┘
                                                     │
                           A 스트리밍 입력            │ B 고정
                           (int8)                    │ (int8)
                           ↓                         ↓
                     ┌─────────────────────────────────┐
                     │        Systolic Array           │
                     │   result += A[i][k] × B[k][j]   │
                     │            (int32)              │
                     └────────────────┬────────────────┘
                                      │
                                      ↓
                            ┌─────────────────┐
                            │   Accumulator   │
                            │    (int32)      │
                            └────────┬────────┘
                                     │ MVOUT
                                     ↓
                              ┌──────────────┐
                              │ Scale + Act  │
                              │  (int32→int8)│
                              └──────┬───────┘
                                     │
                                     ↓
                              ┌──────────────┐
                              │  DRAM (C)    │
                              └──────────────┘
```

---

## 참고

- **API**: `software/gemmini-rocc-tests/include/gemmini.h`
- **모델**: `software/libgemmini/gemmini.cc`
- **하드웨어**: `src/main/scala/gemmini/cont`
