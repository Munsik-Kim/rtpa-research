# DIAG-W W1-OS: a negative result for output-stratified weight allocation

2026-09-23 · **NO_ADVANTAGE_IN_W1_OS** · This specific design is closed.

**English summary.** At the same eight VJPs per calibration document and the
same packed-candidate byte budget, the tested four-output-group/two-probe
estimator did not improve weight allocation over either registered diagonal
control. On a fresh 32-document panel, the primary seed's mean reference-KL was
5.21% higher. One sensitivity seed selected the diagonal controls' identical
map; the other repeated the primary map. We discontinue this specific design,
not response-aware quantization as a whole. Recurrent-state DIAG is a separate
study. These are decoded-BF16 weight-insertion results, not packed-kernel or
deployment benchmarks.

## 결론: 무엇을 접는가?

문서당 8회 역전파와 동일 packed bytes에서, 출력을 네 그룹으로 나눈
`DW_OS4x2`가 강한 대각 기준보다 좋은 weight 조합을 제공한다는 가설은
이번 시험에서 지지되지 않았다. 주 seed는 오히려 평균 KL이 높았고,
세 seed 중 두 대각 기준보다 유리한 방향의 seed는 **0/3**이었다.

따라서 **현재 OS4×2 추정 설계의 개발·확대를 종료한다**. 좋은 seed만 골라
대표 결과로 삼거나 그룹 수·probe 수·seed를 계속 바꾸는 탐색을 이어가지 않는다.
이 결정은 모든 출력 응답 기반 방법의 불가능성 증명이 아니며,
기존 state-DIAG의 결과나 과거 weight W1의 `NO_ADVANTAGE_IN_W1`을 바꾸지 않는다.

## 시험한 변경 하나

- 모델: Qwen/Qwen3-1.7B-Base, revision
  `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
- 선택 단위: layer 3/14/24의 `q_proj`, `o_proj`, `up_proj`, `down_proj`, 총 12개 tensor.
- 동일한 기존 Q2_K/Q4_K packed 후보를 BF16으로 복원하여 삽입했다.
  나머지 weight는 원본 BF16이다. 재양자화·clipping·학습·새 quantizer는 없었다.
- CAL: 기존 16문서(code 8/technical 8). TEST: 새 32문서(code 16/technical 16),
  도메인별 8개 source family × 2문서. 문서별 512 tokens, scored logits `[16,511)`.
- 모든 4,096개 조합 중 동일 upgrade cost인 136개만 비교했다. 12개 tensor의
  배분 payload는 모든 방법에서 **38,928,384 B**, GGUF 파일 합은 **38,930,240 B**다.
  이는 실제 실행 중 전체 모델 VRAM이 아니다. ALL_LOW/HIGH는 다른 예산의 보조 anchor다.

기존 pooled 추정기는 495개 출력을 한 scalar에 합친 독립 probe 8개를 사용한다.
새 추정기는 `g(u)=(u-16) mod 4`로 출력 위치를 고정 분할하고 그룹당 2개 probe를
사용한다. 그룹 크기는 124/124/124/123, 두 추정기 모두 문서당 8 VJP다.

각 signed contraction을 `a`라 하면 pooled 점수는 `sum(a²)/16`, OS는
`sum(a²)/4`다. 각 VJP의 scalar에는 공통 `1/sqrt(495)`가 이미 들어 있다.
그룹 간 **표집 교차항**은 제거되지만, 그룹 내부 표집 항과 weight의 전체 512개
입력 사용 시점 간 결합은 남는다. 이것은 `DW_NO_TIME`이 아니다. 그룹당 probe가
8개에서 2개로 줄어들므로 같은 비용에서 분산이 항상 감소한다는 보장은 없다.

두 공동 주 대조는 `B_OUTDIAG_POOL_R8`와 `B_OUTDIAG_OS4x2`다. 각각 같은 pooled/OS
VJP의 channel-wise squared contraction을 사용하므로 추가 backward는 없다.
Pooled RESP는 별도 진단 비교군이며, 주 대조를 사후에 바꾸지 않았다.
CAL만으로 map을 만들고 TEST 실행 전에 모두 동결했다.

## 새 TEST의 결과

`R = 1 − mean(KL_OS)/mean(KL_baseline)`이며 양수만 OS 개선이다.
KL 방향은 `reference || method`, NLL 차이는 `OS − baseline`이다.
점 추정은 문서 동일 가중이며 단위는 nat/token이다.

| 주 seed s0 | OS4×2 | 각 대각 기준 | 차이 / 구간 |
|---|---:|---:|---|
| 평균 KL | 0.0289073586 | 0.0274766415 | +0.0014307171 |
| 상대 개선 R | — | — | **−5.2070%**, 97.5% CI **[−9.3211%, −1.3692%]** |
| ΔNLL | — | — | +0.0012111686, 97.5% CI [−0.0025458558, +0.0054239880] |
| 문서 승/무/패 | — | — | **13 / 0 / 19** |

두 대각 기준은 실제로 같은 map을 골랐다. 따라서 위의 두 비교는 등록된 두 대조를
모두 보고한 것이지만, 서로 독립적인 두 번의 재현은 아니다.

| 고정 seed | OS가 Q4로 보호한 tensor | 대각 기준 대비 R | 승/무/패 |
|---|---|---:|---|
| s0: 202609230101 (유일한 주 판정) | 3.up, 3.down | −5.2070% | 13/0/19 |
| s1: 202609230102 | 3.up, 24.down | 0% (같은 map) | 0/32/0 |
| s2: 202609230103 | 3.up, 3.down | −5.2070% | 13/0/19 |

`up/down`은 `model.layers.<layer>.mlp.<name>_proj.weight`의 축약이다.
두 대각 기준은 모든 seed에서 3.up/24.down을 선택했다. 나머지는 Q2다.
전체 tensor 이름, pooled map과 anchor까지 포함한 기록은
[precision_maps.json](../results/diag_w_os_public_summary/precision_maps.json)에 있다.
동일 map의 forward는 중복 실행하지 않았다. s1의 0 차이·0 폭 구간은 같은 실행
결과를 가리키는 identity이며 모집단의 동등성이 확정됐다는 뜻이 아니다.

Pooled RESP 대비 R은 s0 −0.2280%, s1 +4.9493%, s2 −3.4682%였다.
s1에서 pooled를 이겼다는 관측으로 대각 기준에 대한 실패를 대체하지 않는다.
아홉 비교의 원정밀도 값과 95%/97.5% 구간은
[primary_comparison.json](../results/diag_w_os_public_summary/primary_comparison.json)에 있다.

통계는 code/technical 내 source-family paired bootstrap 10,000회,
PCG64 seed `202609230777`, linear quantiles다. 같은 family의 두 문서를 함께
뽑고 모든 방법에 같은 draw를 적용했다. 두 공동 주 비교의 97.5% 구간은
명목 Bonferroni 처리이며, 소수 16 families에서 정확한 모집단 coverage를 보장하지 않는다.
문서×seed를 독립 n=96으로 세지 않았다.

등록된 확대 기준은 두 대각 비교 모두 R≥5%, 97.5% 하한>0,
ΔNLL 97.5% 상한≤+0.01, 각 기준 대비 최소 2/3 seed의 유리한 KL 방향이었다.
NLL 조건은 충족했지만 KL·seed 조건은 충족하지 못했다.
[원래 등록 문서](../results/diag_w_os_public_summary/PROTOCOL.md)와
[원래 판정](../results/diag_w_os_public_summary/decision.json)은 그대로 보존했다.

## 반대 관측도 남긴다

주 seed에서 모든 문서·지표가 나빠진 것은 아니다. 32문서 중 13문서,
16 source families 중 6개 family의 평균 KL은 OS가 더 낮았다.
beets, jmespath, oauthlib, paramiko, bokeh, dask가 해당한다.

문서별 tail 지표의 평균은 OS에서 일부 낮았다:
p99 KL 0.2282791 vs 0.2341452, top-1% 평균 KL 0.3970551 vs 0.4033638,
최대 KL 0.6842745 vs 0.7157358. 반면 late KL 평균은 0.02739994 vs 0.02662205로
높았다. Tail의 유리한 점 추정으로 주 평균 KL 판정을 바꾸거나 새 목표를 고르지 않았다.
모든 문서·source·tail 값은 [문서별 CSV](../results/diag_w_os_public_summary/policy_document_metrics.csv)에 있다.

## 비용과 검증 범위

각 추정기는 문서당 8 VJP, 전체 세 seed/CAL에서 각각 384 main VJP를 사용했다.
POOL/OS 통합 수집 시간은 196.352/196.116초, OS/POOL 비율은 0.9988이었다.
이는 두 점수를 함께 모으고 hash 검사를 포함한 실행의 비용이며, 최적화된 독립 scorer나
serving latency 비교가 아니다. [비용 원기록](../results/diag_w_os_public_summary/cost_summary.json).

전체 gate+CAL+TEST는 1,010 forward / 517,120 input positions / 781 VJP였다.
GPU worker 누적 549.438초, peak allocated 7.710 GiB, sampled process-tree RSS
3.755 GiB, GPU retry 0이었다. 이번 공개 작업에서는 새 모델 계산을 하지 않았다.

원래 R8 exact replay, reference/candidate/복원 raw bytes 연결을 통과했다.
그룹 contraction 재결합의 vector-relative L2는 0.013672로 등록 gate 0.05 아래였다.
이를 서로 다른 BF16 AD 경로의 bitwise 일치라고 부르지 않는다.
로컬 저장 자료 검산은 46,402 assertions / 27,609 FP64 값에서 통과했다.
[verification.json](../results/diag_w_os_public_summary/verification.json)은 그때의
변경 없는 receipt다. 이번 작은 공개 묶음이 그 전체 검산이나 GPU 미분을 다시
실행했다는 뜻은 아니다. 공개 묶음의 별도 CPU 재집계 범위는 아래에 한정된다.

## 무엇을 알았고, 무엇을 알지 못하는가?

관측한 사실은 이 고정 추정기의 **동일-byte 실제 배분 결과가 두 강한 대각 기준보다
좋지 않았다는 것**이다. 이전 LR의 사후 출력별 분해는 설계 동기였을 뿐,
같은 8 VJP 예산의 새 추정기 성능을 보장하지 않았다.

어떤 표집 항이 손실의 몇 %를 인과적으로 만들었는지, 다른 partition·probe 예산이면
어떨지까지 확인한 것은 아니다. 이번 결과만으로 모든 국소 응답·future-response
기법이 효과 없거나 대각 기준이 보편적으로 최적이라고 결론내리지 않는다.

12개 tensor 외에는 BF16이다. 전체 weight 압축, packed low-bit matmul, task accuracy,
자유 생성 품질, serving 속도, 전체 VRAM 절감, 대형 모델 10~12GB 배포는 미측정이다.
Novelty는 `NOT_ESTABLISHED`이며 선행 출력-aware 방법 전체 대비 우위를 주장하지 않는다.

TEST는 기록된 과거 패널의 알려진 source family를 제외했지만, 제공되지 않은 과거
원문까지 near-duplicate 부재를 확인하지 못했다. Pretraining contamination은 UNKNOWN이다.
원래 W1 serializer adapter, W1-D schema/doc-first 수정, CAL 축 오류/GPU 접근 예외,
LR R1 numerical stop/R2 mass guard 이력을 삭제하거나 최초 clean PASS로 바꾸지 않았다.
OS의 등록 전 source 준비/loader 및 CPU-test logging 수정도 기존 기록에 남아 있다.
이번 결론은 기존 state 연구나 기본 runtime/backend를 변경하지 않는다.

## 공개 범위와 재집계

[작은 공개 근거 묶음](../results/diag_w_os_public_summary/README.md)은 판정, 원래 등록 문서,
선택 map, bytes ledger, 32문서 scalar 및 source hashes를 포함한다.
기존 NumPy 환경에서 repository 루트에서 실행한다:

```bash
python scripts/recheck_diag_w_os_public.py
```

이 명령은 다운로드·Torch·모델 실행 없이 문서 평균/승패/paired CI/예산·동일 map과
부정 판정 조건을 재집계한다. 전체 scorer, packed 후보, token 원관측 또는 GPU 실험의
독립 재현 명령은 아니다. 새 추론 policy를 생성하지 않는다.

로컬 결과 commit은 `ad03e0339a6357d3c3b1d853a4adfa21705f2a7d`다.
연구 브랜치 전체는 이 공개 변경에 포함하지 않으므로 이 hash의 GitHub 접근을 보장하지 않는다.
모델·후보·원문·큰 배열·환경과 141,624,025-byte review ZIP은 업로드하지 않았다.
해당 로컬 ZIP SHA-256은
`8b09f03eeb3f358deb9447ddb149938db29af9eb873334e9ac4edc67c68d2fb6`이다.
정확한 원본/공개 파일 hash 및 변환은
[provenance.json](../results/diag_w_os_public_summary/provenance.json)에 기록했다.

**다음 실행은 없다.** 이 방향의 범위를 닫고 부정 결과를 보존한다. W2/W3나 다른
후보 연구를 진행하려면 별도 설계·승인이 필요하다.
