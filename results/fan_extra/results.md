# Fan session: exploratory follow-up analyses (post hoc)

Gates from fan distance 1.0 m on the radar axis (as in the primary analysis).

## 1. CFAR fairness: detection AUC

| Method | front off | front on | side off | side on |
|---|---|---|---|---|
| SR + global | 0.974 | 0.994 | 0.986 | 0.943 |
| SR + PCEN tau=1s | 0.973 | 0.999 | 0.995 | 0.955 |
| SR + clutter-map | 0.977 | 0.999 | 0.993 | 0.955 |
| CA 2x2/8x4 (paper) | 0.964 | 0.989 | 0.988 | 0.791 |
| CA 1x1/4x4 | 0.959 | 0.991 | 0.985 | 0.755 |
| CA 2x2/4x8 | 0.979 | 0.985 | 0.977 | 0.746 |
| CA 2x2/16x4 | 0.960 | 0.993 | 0.988 | 0.886 |
| CA Doppler-only 0x2/0x8 | 0.976 | 0.970 | 0.956 | 0.941 |
| GO-CFAR 2x2/8x4 | 0.968 | 0.987 | 0.987 | 0.742 |
| SO-CFAR 2x2/8x4 | 0.958 | 0.971 | 0.986 | 0.802 |
| OS-CFAR k=0.75 | 0.970 | 0.980 | 0.985 | 0.788 |
| OS-CFAR k=0.5 | 0.946 | 0.875 | 0.932 | 0.799 |

Tape-distance gates (fan 0.8 m), side, fan on: SR + PCEN tau=1s 0.985, CA 2x2/8x4 (paper) 0.928, OS-CFAR k=0.75 0.905, GO-CFAR 2x2/8x4 0.888, SO-CFAR 2x2/8x4 0.914

## 2. PCEN alpha and tau (AUC side on / fan FA increase side [dB])

| Setting | AUC side on | AUC front on | FA increase front | FA increase side |
|---|---|---|---|---|
| pcen_a0.0 | 0.943 | 0.994 | +21.3 | +29.6 |
| pcen_a0.25 | 0.956 | 0.997 | +19.8 | +29.0 |
| pcen_a0.5 | 0.966 | 0.999 | +16.8 | +27.3 |
| pcen_a0.75 | 0.969 | 0.999 | +10.2 | +21.5 |
| pcen_a0.9 | 0.964 | 1.000 | +3.2 | +9.8 |
| pcen_a0.98 | 0.955 | 0.999 | -1.0 | -1.7 |
| pcen_a1.0 | 0.955 | 0.999 | -1.8 | -4.6 |
| pcen_t0.05 | 0.975 | 1.000 | +7.2 | +14.3 |
| pcen_t0.1 | 0.962 | 0.999 | +1.4 | +5.3 |
| pcen_t0.13 | 0.955 | 1.000 | -0.0 | +2.5 |
| pcen_t0.25 | 0.951 | 0.999 | -1.6 | -1.4 |
| pcen_t0.5 | 0.952 | 0.999 | -2.0 | -2.5 |
| pcen_t1.0 | 0.955 | 0.999 | -1.0 | -1.7 |
| pcen_t2.0 | 0.960 | 1.000 | +0.5 | +0.4 |
| pcen_t5.0 | 0.962 | 0.999 | +2.3 | +4.8 |
| pcen_t10.0 | 0.961 | 0.999 | +3.4 | +8.5 |
| clut_t0.05 | 0.884 | 0.976 | -2.4 | -4.9 |
| clut_t0.1 | 0.928 | 0.993 | -2.7 | -6.6 |
| clut_t0.13 | 0.936 | 0.996 | -2.5 | -7.4 |
| clut_t0.25 | 0.950 | 0.999 | -3.0 | -7.4 |
| clut_t0.5 | 0.953 | 0.999 | -3.1 | -6.6 |
| clut_t1.0 | 0.955 | 0.999 | -1.8 | -4.6 |
| clut_t2.0 | 0.959 | 1.000 | -0.4 | -2.0 |
| clut_t5.0 | 0.960 | 0.999 | +1.6 | +2.8 |
| clut_t10.0 | 0.960 | 0.999 | +2.8 | +6.9 |
| pcen_d0.5_r0.25 | 0.955 | 0.999 | -1.0 | -1.7 |
| pcen_d10_r1 | 0.955 | 0.999 | -1.0 | -1.7 |

## 3. Pd at fixed window false-alarm rate (side, fan on)

| Method | Pd@1% | Pd@5% | Pd@10% |
|---|---|---|---|
| SR + global | 0.55 | 0.80 | 0.88 |
| CA 2x2/8x4 (paper) | 0.30 | 0.42 | 0.52 |
| OS-CFAR k=0.75 | 0.25 | 0.38 | 0.40 |
| SR + clutter-map | 0.45 | 0.97 | 0.97 |
| SR + PCEN tau=1s | 0.50 | 0.97 | 0.97 |

## 4. AUC per gesture (side, fan on)

| Method | swipe | push | circle |
|---|---|---|---|
| SR + global | 0.916 | 0.957 | 0.954 |
| CA 2x2/8x4 (paper) | 0.695 | 0.949 | 0.729 |
| OS-CFAR k=0.75 | 0.729 | 0.957 | 0.679 |
| SR + clutter-map | 0.887 | 0.997 | 0.981 |
| SR + PCEN tau=1s | 0.890 | 0.997 | 0.979 |

## 5. Cluster bootstrap (clusters of 5 repetitions)

| Contrast | clusters | 95% CI | p one-sided |
|---|---|---|---|
| front_on: PCEN vs CA 2x2/8x4 (paper) | 18 | [+0.000, +0.033] | 0.0305 |
| front_on: PCEN vs SR + global | 18 | [+0.000, +0.014] | 0.0155 |
| front_on: PCEN vs OS-CFAR k=0.75 | 18 | [+0.005, +0.038] | 0.0010 |
| side_on: PCEN vs CA 2x2/8x4 (paper) | 18 | [+0.050, +0.295] | 0.0015 |
| side_on: PCEN vs SR + global | 18 | [-0.024, +0.049] | 0.2659 |
| side_on: PCEN vs OS-CFAR k=0.75 | 18 | [+0.050, +0.300] | 0.0010 |

## 6. Runtime per 64x64 frame (numpy, this machine)

- PCEN (streaming): 0.062 ms
- clutter-map (streaming): 0.018 ms
- CA 2x2/8x4 (paper): 1.814 ms
- GO-CFAR 2x2/8x4: 1.960 ms
- OS-CFAR k=0.75: 19.058 ms