# Swing fidelity and Jensen-gap matrix

Source: `../results_raw`. Converged checkpoints: **111**; scored: **111**; missing: **0**; malformed: **0**.

All summaries are median and full min–max over seeds. Gate counts are reported explicitly; a failed seed is never dropped. `n/e` means the model tracked no swing band at median thermal MAE <= 5 degC, so the spectral-bias gate was not evaluable.

## 1. Cell summary

| cell | architecture | cascade | baseline | scored | gate (pass/fail/n/e) | median swing ratio [min–max] | median thermal MAE degC [min–max] |
|---|---|---|---|---|---|---|---|
| `cod` | PI-DeepONet | True | True | 6 | 6/0/0 | 1.0103 [1.0055–1.0176] | 0.3309 [0.2366–0.4205] |
| `cod_bounded_correction` | PI-DeepONet | True | True | 1 | 0/0/1 | 1.4951 [1.4951–1.4951] | 15.5426 [15.5426–15.5426] |
| `cod_no_baseline` | PI-DeepONet | True | False | 6 | 1/5/0 | 0.9970 [0.8935–1.0321] | 1.4831 [1.1551–2.0715] |
| `fno_baseline_in_cascade` | FNO | True | True | 7 | 7/0/0 | 1.0356 [1.0310–1.0465] | 0.3137 [0.2208–0.5165] |
| `fno_baseline_monolithic` | FNO | False | True | 7 | 7/0/0 | 1.0265 [1.0169–1.0313] | 0.4119 [0.3295–0.5710] |
| `fno_in_cascade` | FNO | True | False | 7 | 0/7/0 | 0.6886 [0.5568–0.7912] | 4.2355 [2.8418–5.8710] |
| `fno_monolithic` | FNO | False | False | 7 | 0/7/0 | 0.7216 [0.6253–0.7616] | 4.0398 [3.7498–5.8392] |
| `mionet_baseline_in_cascade` | MIONet | True | True | 7 | 7/0/0 | 1.1275 [1.0212–1.3245] | 0.8018 [0.4158–1.1651] |
| `mionet_baseline_monolithic` | MIONet | False | True | 7 | 7/0/0 | 1.1286 [1.0339–1.2004] | 0.9105 [0.6226–1.0626] |
| `mionet_in_cascade` | MIONet | True | False | 7 | 1/6/0 | 0.9700 [0.9465–0.9819] | 1.0969 [0.5700–2.0062] |
| `mionet_monolithic` | MIONet | False | False | 7 | 3/4/0 | 0.9661 [0.9200–0.9896] | 1.0923 [0.8102–1.4692] |
| `pideeponet_baseline_monolithic` | PI-DeepONet | False | True | 7 | 6/1/0 | 1.0674 [1.0560–1.0788] | 0.6011 [0.4909–0.6843] |
| `pideeponet_monolithic` | PI-DeepONet | False | False | 7 | 1/6/0 | 0.7671 [0.6810–0.8161] | 3.1983 [2.4398–4.1755] |
| `sdeeponet_baseline_in_cascade` | S-DeepONet | True | True | 7 | 7/0/0 | 1.1220 [1.0789–1.2673] | 0.8853 [0.5955–5.0818] |
| `sdeeponet_baseline_monolithic` | S-DeepONet | False | True | 7 | 6/1/0 | 1.1738 [1.0325–1.2438] | 0.9815 [0.6686–1.1830] |
| `sdeeponet_in_cascade` | S-DeepONet | True | False | 7 | 5/2/0 | 0.9939 [0.9282–1.0405] | 0.6894 [0.3012–1.2397] |
| `sdeeponet_monolithic` | S-DeepONet | False | False | 7 | 3/4/0 | 0.9519 [0.9140–1.0004] | 0.6146 [0.1714–1.4791] |

## 2. Jensen-gap preservation

Median of each seed's median predicted/true Jensen-gap ratio over live cases. One is exact preservation; below one means gap lost.

| cell | `c_H2` | `c_C2H2` | `c_C2H4` | `c_CO` | `c_CO2` | `DP` |
|---|---|---|---|---|---|---|
| `cod` | 1.0065 [1.0029–1.0093] | 1.0150 [1.0064–1.0215] | 1.0094 [1.0042–1.0135] | 1.0037 [1.0018–1.0063] | 1.0027 [1.0012–1.0046] | 1.0079 [1.0036–1.0112] |
| `cod_bounded_correction` | 1.1555 [1.1555–1.1555] | 1.3136 [1.3136–1.3136] | 1.2168 [1.2168–1.2168] | 1.0957 [1.0957–1.0957] | 1.0684 [1.0684–1.0684] | 1.1873 [1.1873–1.1873] |
| `cod_no_baseline` | 0.9919 [0.9673–1.0092] | 0.9770 [0.9351–1.0324] | 0.9865 [0.9533–1.0216] | 0.9952 [0.9796–1.0044] | 0.9960 [0.9850–1.0038] | 0.9894 [0.9606–1.0150] |
| `fno_baseline_in_cascade` | 1.0120 [1.0084–1.0154] | 1.0232 [1.0204–1.0323] | 1.0154 [1.0128–1.0215] | 1.0077 [1.0049–1.0095] | 1.0056 [1.0035–1.0072] | 1.0138 [1.0105–1.0184] |
| `fno_baseline_monolithic` | 1.0118 [1.0049–1.0137] | 1.0291 [1.0119–1.0329] | 1.0180 [1.0073–1.0206] | 1.0064 [1.0030–1.0082] | 1.0045 [1.0021–1.0058] | 1.0149 [1.0061–1.0168] |
| `fno_in_cascade` | 0.8486 [0.8125–0.8713] | 0.7108 [0.6259–0.7548] | 0.7899 [0.7288–0.8210] | 0.9056 [0.8775–0.9189] | 0.9281 [0.9064–0.9399] | 0.8191 [0.7729–0.8443] |
| `fno_monolithic` | 0.8490 [0.8237–0.8644] | 0.7127 [0.6488–0.7249] | 0.7934 [0.7536–0.7978] | 0.9035 [0.8843–0.9153] | 0.9291 [0.9153–0.9376] | 0.8193 [0.7920–0.8316] |
| `mionet_baseline_in_cascade` | 1.0211 [1.0093–1.0450] | 1.0648 [1.0218–1.1352] | 1.0347 [1.0144–1.0733] | 1.0118 [1.0061–1.0249] | 1.0084 [1.0048–1.0177] | 1.0273 [1.0117–1.0580] |
| `mionet_baseline_monolithic` | 1.0293 [1.0208–1.0410] | 1.0792 [1.0500–1.1089] | 1.0474 [1.0310–1.0612] | 1.0172 [1.0129–1.0249] | 1.0119 [1.0091–1.0180] | 1.0371 [1.0260–1.0499] |
| `mionet_in_cascade` | 0.9884 [0.9818–0.9979] | 0.9744 [0.9641–0.9866] | 0.9833 [0.9751–0.9947] | 0.9936 [0.9887–0.9989] | 0.9955 [0.9916–0.9994] | 0.9858 [0.9780–0.9961] |
| `mionet_monolithic` | 0.9811 [0.9576–0.9896] | 0.9581 [0.9104–0.9756] | 0.9735 [0.9403–0.9845] | 0.9886 [0.9729–0.9937] | 0.9918 [0.9797–0.9954] | 0.9779 [0.9490–0.9868] |
| `pideeponet_baseline_monolithic` | 1.0286 [1.0236–1.0372] | 1.0673 [1.0525–1.0895] | 1.0458 [1.0326–1.0589] | 1.0171 [1.0143–1.0225] | 1.0122 [1.0106–1.0163] | 1.0374 [1.0281–1.0476] |
| `pideeponet_monolithic` | 0.9200 [0.8648–0.9412] | 0.8375 [0.7198–0.8737] | 0.8869 [0.8112–0.9269] | 0.9508 [0.9139–0.9651] | 0.9641 [0.9350–0.9754] | 0.9039 [0.8396–0.9370] |
| `sdeeponet_baseline_in_cascade` | 1.0369 [1.0143–1.0871] | 1.0917 [1.0306–1.2436] | 1.0548 [1.0203–1.1391] | 1.0214 [1.0085–1.0507] | 1.0153 [1.0064–1.0368] | 1.0464 [1.0172–1.1120] |
| `sdeeponet_baseline_monolithic` | 1.0387 [1.0293–1.0508] | 1.0998 [1.0706–1.1480] | 1.0633 [1.0450–1.0843] | 1.0233 [1.0170–1.0286] | 1.0174 [1.0131–1.0202] | 1.0507 [1.0374–1.0654] |
| `sdeeponet_in_cascade` | 0.9956 [0.9487–1.0196] | 0.9892 [0.8916–1.0448] | 0.9935 [0.9285–1.0295] | 0.9967 [0.9685–1.0116] | 0.9979 [0.9765–1.0085] | 0.9945 [0.9387–1.0243] |
| `sdeeponet_monolithic` | 0.9855 [0.9597–1.0009] | 0.9685 [0.9138–1.0015] | 0.9792 [0.9440–1.0013] | 0.9909 [0.9756–1.0006] | 0.9933 [0.9822–1.0005] | 0.9824 [0.9508–1.0011] |

## 3. Gate failures by tracked swing band

| cell | band (degC) | scored seeds | tracked seeds | failed seeds | median ratio [min–max], tracked seeds |
|---|---|---|---|---|---|
| `cod` | 1–5 | 6 | 6 | 0 | 1.0141 [1.0021–1.0371] |
| `cod` | 5–10 | 6 | 6 | 0 | 1.0029 [0.9838–1.0134] |
| `cod` | 10–15 | 6 | 6 | 0 | 1.0087 [1.0065–1.0201] |
| `cod` | 15–25 | 6 | 6 | 0 | 1.0106 [1.0069–1.0191] |
| `cod` | 25–200 | 6 | 6 | 0 | 1.0408 [1.0258–1.0542] |
| `cod_bounded_correction` | 1–5 | 1 | 0 | 0 | n/e |
| `cod_bounded_correction` | 5–10 | 1 | 0 | 0 | n/e |
| `cod_bounded_correction` | 10–15 | 1 | 0 | 0 | n/e |
| `cod_bounded_correction` | 15–25 | 1 | 0 | 0 | n/e |
| `cod_bounded_correction` | 25–200 | 1 | 0 | 0 | n/e |
| `cod_no_baseline` | 1–5 | 6 | 6 | 1 | 1.0295 [0.9293–1.0764] |
| `cod_no_baseline` | 5–10 | 6 | 6 | 1 | 1.0461 [0.9020–1.0767] |
| `cod_no_baseline` | 10–15 | 6 | 6 | 1 | 1.0075 [0.9009–1.0931] |
| `cod_no_baseline` | 15–25 | 6 | 6 | 1 | 0.9808 [0.9118–1.0212] |
| `cod_no_baseline` | 25–200 | 6 | 6 | 5 | 0.8895 [0.8197–0.9626] |
| `fno_baseline_in_cascade` | 1–5 | 7 | 7 | 0 | 1.0333 [1.0134–1.0479] |
| `fno_baseline_in_cascade` | 5–10 | 7 | 7 | 0 | 1.0340 [1.0221–1.0404] |
| `fno_baseline_in_cascade` | 10–15 | 7 | 7 | 0 | 1.0374 [1.0270–1.0440] |
| `fno_baseline_in_cascade` | 15–25 | 7 | 7 | 0 | 1.0409 [1.0321–1.0446] |
| `fno_baseline_in_cascade` | 25–200 | 7 | 7 | 0 | 1.0574 [1.0472–1.0663] |
| `fno_baseline_monolithic` | 1–5 | 7 | 7 | 0 | 1.0265 [1.0166–1.0440] |
| `fno_baseline_monolithic` | 5–10 | 7 | 7 | 0 | 1.0188 [1.0098–1.0313] |
| `fno_baseline_monolithic` | 10–15 | 7 | 7 | 0 | 1.0245 [1.0149–1.0293] |
| `fno_baseline_monolithic` | 15–25 | 7 | 7 | 0 | 1.0275 [1.0234–1.0287] |
| `fno_baseline_monolithic` | 25–200 | 7 | 7 | 0 | 1.0525 [1.0475–1.0652] |
| `fno_in_cascade` | 1–5 | 7 | 7 | 5 | 0.8960 [0.6224–0.9858] |
| `fno_in_cascade` | 5–10 | 7 | 7 | 7 | 0.7230 [0.6231–0.8533] |
| `fno_in_cascade` | 10–15 | 7 | 4 | 4 | 0.6829 [0.6801–0.7827] |
| `fno_in_cascade` | 15–25 | 7 | 3 | 3 | 0.7237 [0.6865–0.7932] |
| `fno_in_cascade` | 25–200 | 7 | 0 | 0 | n/e |
| `fno_monolithic` | 1–5 | 7 | 7 | 2 | 0.9912 [0.8039–1.0697] |
| `fno_monolithic` | 5–10 | 7 | 7 | 7 | 0.7884 [0.6790–0.8957] |
| `fno_monolithic` | 10–15 | 7 | 4 | 4 | 0.6917 [0.6567–0.7325] |
| `fno_monolithic` | 15–25 | 7 | 3 | 3 | 0.7421 [0.7199–0.7677] |
| `fno_monolithic` | 25–200 | 7 | 0 | 0 | n/e |
| `mionet_baseline_in_cascade` | 1–5 | 7 | 7 | 0 | 2.1710 [1.0070–2.6404] |
| `mionet_baseline_in_cascade` | 5–10 | 7 | 7 | 0 | 1.3884 [1.0247–1.5431] |
| `mionet_baseline_in_cascade` | 10–15 | 7 | 7 | 0 | 1.3332 [1.0147–1.4602] |
| `mionet_baseline_in_cascade` | 15–25 | 7 | 7 | 0 | 1.0199 [1.0097–1.0356] |
| `mionet_baseline_in_cascade` | 25–200 | 7 | 7 | 0 | 1.1113 [1.0604–1.1318] |
| `mionet_baseline_monolithic` | 1–5 | 7 | 7 | 0 | 2.1385 [1.0224–2.3756] |
| `mionet_baseline_monolithic` | 5–10 | 7 | 7 | 0 | 1.3972 [1.0499–1.5296] |
| `mionet_baseline_monolithic` | 10–15 | 7 | 7 | 0 | 1.3296 [1.0209–1.3914] |
| `mionet_baseline_monolithic` | 15–25 | 7 | 7 | 0 | 1.0282 [1.0188–1.0412] |
| `mionet_baseline_monolithic` | 25–200 | 7 | 7 | 0 | 1.1153 [1.0715–1.1237] |
| `mionet_in_cascade` | 1–5 | 7 | 7 | 3 | 0.9949 [0.8987–1.0972] |
| `mionet_in_cascade` | 5–10 | 7 | 7 | 2 | 1.0009 [0.9417–1.0902] |
| `mionet_in_cascade` | 10–15 | 7 | 7 | 0 | 0.9888 [0.9527–1.0506] |
| `mionet_in_cascade` | 15–25 | 7 | 7 | 3 | 0.9587 [0.9363–0.9815] |
| `mionet_in_cascade` | 25–200 | 7 | 7 | 3 | 0.9563 [0.8643–0.9686] |
| `mionet_monolithic` | 1–5 | 7 | 7 | 2 | 0.9777 [0.7702–1.0950] |
| `mionet_monolithic` | 5–10 | 7 | 7 | 2 | 0.9737 [0.8807–1.0539] |
| `mionet_monolithic` | 10–15 | 7 | 7 | 2 | 0.9644 [0.9261–1.0102] |
| `mionet_monolithic` | 15–25 | 7 | 7 | 3 | 0.9514 [0.9031–1.0044] |
| `mionet_monolithic` | 25–200 | 7 | 7 | 3 | 0.9575 [0.8764–0.9876] |
| `pideeponet_baseline_monolithic` | 1–5 | 7 | 7 | 1 | 1.2898 [0.9391–1.8973] |
| `pideeponet_baseline_monolithic` | 5–10 | 7 | 7 | 0 | 1.1104 [1.0522–1.2528] |
| `pideeponet_baseline_monolithic` | 10–15 | 7 | 7 | 0 | 1.0682 [1.0377–1.0780] |
| `pideeponet_baseline_monolithic` | 15–25 | 7 | 7 | 0 | 1.0449 [1.0398–1.0570] |
| `pideeponet_baseline_monolithic` | 25–200 | 7 | 7 | 0 | 1.0615 [1.0420–1.0716] |
| `pideeponet_monolithic` | 1–5 | 7 | 7 | 0 | 1.1516 [0.9691–1.3024] |
| `pideeponet_monolithic` | 5–10 | 7 | 7 | 6 | 0.9224 [0.8388–0.9518] |
| `pideeponet_monolithic` | 10–15 | 7 | 7 | 6 | 0.8568 [0.6915–0.9592] |
| `pideeponet_monolithic` | 15–25 | 7 | 0 | 0 | n/e |
| `pideeponet_monolithic` | 25–200 | 7 | 0 | 0 | n/e |
| `sdeeponet_baseline_in_cascade` | 1–5 | 7 | 7 | 0 | 2.1168 [1.2192–2.3437] |
| `sdeeponet_baseline_in_cascade` | 5–10 | 7 | 7 | 0 | 1.4273 [1.1046–1.5076] |
| `sdeeponet_baseline_in_cascade` | 10–15 | 7 | 6 | 0 | 1.3065 [1.0969–1.3895] |
| `sdeeponet_baseline_in_cascade` | 15–25 | 7 | 6 | 0 | 1.0344 [1.0181–1.0473] |
| `sdeeponet_baseline_in_cascade` | 25–200 | 7 | 6 | 0 | 1.1098 [1.0994–1.1285] |
| `sdeeponet_baseline_monolithic` | 1–5 | 7 | 7 | 1 | 2.2561 [0.9378–2.5106] |
| `sdeeponet_baseline_monolithic` | 5–10 | 7 | 7 | 0 | 1.5100 [1.0028–1.5766] |
| `sdeeponet_baseline_monolithic` | 10–15 | 7 | 7 | 0 | 1.3961 [1.0293–1.4645] |
| `sdeeponet_baseline_monolithic` | 15–25 | 7 | 7 | 0 | 1.0408 [1.0200–1.0476] |
| `sdeeponet_baseline_monolithic` | 25–200 | 7 | 7 | 0 | 1.1235 [1.0985–1.1313] |
| `sdeeponet_in_cascade` | 1–5 | 7 | 7 | 2 | 1.0115 [0.8680–1.0319] |
| `sdeeponet_in_cascade` | 5–10 | 7 | 7 | 1 | 0.9802 [0.8965–1.0511] |
| `sdeeponet_in_cascade` | 10–15 | 7 | 7 | 1 | 0.9899 [0.9472–1.0581] |
| `sdeeponet_in_cascade` | 15–25 | 7 | 7 | 1 | 0.9930 [0.9374–1.0350] |
| `sdeeponet_in_cascade` | 25–200 | 7 | 7 | 1 | 0.9903 [0.9082–1.0362] |
| `sdeeponet_monolithic` | 1–5 | 7 | 7 | 2 | 0.9775 [0.8016–1.0125] |
| `sdeeponet_monolithic` | 5–10 | 7 | 7 | 3 | 0.9590 [0.9145–0.9974] |
| `sdeeponet_monolithic` | 10–15 | 7 | 7 | 0 | 0.9720 [0.9510–1.0008] |
| `sdeeponet_monolithic` | 15–25 | 7 | 7 | 4 | 0.9499 [0.8862–1.0006] |
| `sdeeponet_monolithic` | 25–200 | 7 | 7 | 3 | 0.9519 [0.9030–1.0005] |

## 4. Integrity

**PASS** — every converged checkpoint has one readable swing JSON on the frozen distribution.
