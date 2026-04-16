# Table 1 Companion: Median + IQR

This companion table keeps the same `HW vs. LLM transfer` layout,
but summarizes each detector using stage-level `median [IQR]` instead of
`overall ± std`.

Interpretation notes:

- Each cell is formatted as `median [IQR]`.
- Here `IQR = Q3 - Q1` across the available stage-level metrics.
- `Δ = LLM - HW`, so a negative value means the detector is weaker on LLM-generated content.
- This table is a robustness-oriented view across stages, not a raw-sample overall aggregation.

- `Academic` paired detectors for stage-robust transfer summary: `n = 6`.
- `Industrial` paired detectors for stage-robust transfer summary: `n = 3`.

<table>
  <thead>
    <tr>
      <th rowspan="2">Group</th>
      <th rowspan="2">Detector</th>
      <th colspan="3">Recall</th>
      <th colspan="3">F2</th>
      <th colspan="3">MCC</th>
    </tr>
    <tr>
      <th>HW</th>
      <th>LLM</th>
      <th>Δ</th>
      <th>HW</th>
      <th>LLM</th>
      <th>Δ</th>
      <th>HW</th>
      <th>LLM</th>
      <th>Δ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Academic</strong></td>
      <td>ml_watermark_logreg</td>
      <td>0.8614 [0.0119]</td>
      <td>0.8822 [0.1224]</td>
      <td>0.0234 [0.1257]</td>
      <td>0.7514 [0.0154]</td>
      <td>0.7828 [0.0466]</td>
      <td>0.0354 [0.0778]</td>
      <td>-0.0310 [0.0175]</td>
      <td>0.0867 [0.3123]</td>
      <td>0.1147 [0.2879]</td>
    </tr>
    <tr>
      <td></td>
      <td>pimref</td>
      <td>0.0406 [0.0059]</td>
      <td>0.0195 [0.0324]</td>
      <td>-0.0206 [0.0282]</td>
      <td>0.0501 [0.0072]</td>
      <td>0.0243 [0.0400]</td>
      <td>-0.0253 [0.0346]</td>
      <td>0.0851 [0.0165]</td>
      <td>0.0921 [0.0608]</td>
      <td>-0.0088 [0.0647]</td>
    </tr>
    <tr>
      <td></td>
      <td>scamllm</td>
      <td>0.8748 [0.0046]</td>
      <td>0.8009 [0.1377]</td>
      <td>-0.0733 [0.1372]</td>
      <td>0.8538 [0.0092]</td>
      <td>0.7712 [0.0716]</td>
      <td>-0.0725 [0.0773]</td>
      <td>0.6309 [0.0249]</td>
      <td>0.4627 [0.1900]</td>
      <td>-0.1270 [0.1833]</td>
    </tr>
    <tr>
      <td></td>
      <td>securenet_llama</td>
      <td>0.8299 [0.0169]</td>
      <td>0.6828 [0.2997]</td>
      <td>-0.1529 [0.2854]</td>
      <td>0.8525 [0.0172]</td>
      <td>0.7186 [0.1253]</td>
      <td>-0.1299 [0.1076]</td>
      <td>0.7801 [0.0239]</td>
      <td>0.6274 [0.3008]</td>
      <td>-0.1329 [0.2780]</td>
    </tr>
    <tr>
      <td></td>
      <td>t5phishing</td>
      <td>1.0000 [0.0000]</td>
      <td>1.0000 [0.0000]</td>
      <td>0.0000 [0.0000]</td>
      <td>0.8333 [0.0000]</td>
      <td>0.8333 [0.0000]</td>
      <td>0.0000 [0.0000]</td>
      <td>0.0000 [0.0000]</td>
      <td>0.0000 [0.0000]</td>
      <td>0.0000 [0.0000]</td>
    </tr>
    <tr>
      <td></td>
      <td>xgboost</td>
      <td>0.6487 [0.0097]</td>
      <td>0.6372 [0.1823]</td>
      <td>-0.0027 [0.1819]</td>
      <td>0.6591 [0.0079]</td>
      <td>0.6549 [0.1561]</td>
      <td>-0.0138 [0.1759]</td>
      <td>0.3436 [0.0365]</td>
      <td>0.2277 [0.1633]</td>
      <td>-0.1233 [0.2160]</td>
    </tr>
    <tr>
      <td></td>
      <td><strong>Academic median</strong></td>
      <td>0.8457 [0.0078]</td>
      <td>0.7418 [0.1300]</td>
      <td>-0.0117 [0.1314]</td>
      <td>0.7924 [0.0085]</td>
      <td>0.7449 [0.0591]</td>
      <td>-0.0195 [0.0775]</td>
      <td>0.2143 [0.0207]</td>
      <td>0.1599 [0.1766]</td>
      <td>-0.0661 [0.1997]</td>
    </tr>
    <tr>
      <td></td>
      <td><strong>Academic IQR</strong></td>
      <td>0.1774 [0.0064]</td>
      <td>0.2132 [0.1163]</td>
      <td>0.0595 [0.1181]</td>
      <td>0.1655 [0.0065]</td>
      <td>0.1091 [0.0702]</td>
      <td>0.0572 [0.0549]</td>
      <td>0.5378 [0.0079]</td>
      <td>0.3159 [0.1866]</td>
      <td>0.1239 [0.1682]</td>
    </tr>
    <tr>
      <td><strong>Industrial</strong></td>
      <td>email_phishing_detection_v3</td>
      <td>-</td>
      <td>0.8947 [0.1631]</td>
      <td>-</td>
      <td>-</td>
      <td>0.7179 [0.1362]</td>
      <td>-</td>
      <td>-</td>
      <td>0.4536 [0.1916]</td>
      <td>-</td>
    </tr>
    <tr>
      <td></td>
      <td>llm_guard</td>
      <td>0.4194 [0.0183]</td>
      <td>0.5325 [0.3241]</td>
      <td>-0.0320 [0.1507]</td>
      <td>0.4464 [0.0132]</td>
      <td>0.5068 [0.2494]</td>
      <td>-0.0390 [0.1081]</td>
      <td>0.0639 [0.0180]</td>
      <td>-0.0107 [0.3419]</td>
      <td>-0.1703 [0.3141]</td>
    </tr>
    <tr>
      <td></td>
      <td>oopspam</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td></td>
      <td>phishing_email_agent</td>
      <td>0.2188 [0.0057]</td>
      <td>0.3361 [0.1380]</td>
      <td>0.1166 [0.1028]</td>
      <td>0.2566 [0.0067]</td>
      <td>0.3683 [0.1477]</td>
      <td>0.1118 [0.1094]</td>
      <td>0.2348 [0.0296]</td>
      <td>0.3118 [0.1175]</td>
      <td>-0.0014 [0.1004]</td>
    </tr>
    <tr>
      <td></td>
      <td>pyrit_blocklist</td>
      <td>-</td>
      <td>0.1625 [0.1280]</td>
      <td>-</td>
      <td>-</td>
      <td>0.1920 [0.1455]</td>
      <td>-</td>
      <td>-</td>
      <td>0.1487 [0.0688]</td>
      <td>-</td>
    </tr>
    <tr>
      <td></td>
      <td>pyrit_original</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td></td>
      <td>spamassassin</td>
      <td>0.4148 [0.0160]</td>
      <td>0.1903 [0.2059]</td>
      <td>-0.2251 [0.2023]</td>
      <td>0.4677 [0.0166]</td>
      <td>0.2266 [0.1754]</td>
      <td>-0.2423 [0.1751]</td>
      <td>0.4665 [0.0415]</td>
      <td>0.2388 [0.0969]</td>
      <td>-0.2838 [0.1354]</td>
    </tr>
    <tr>
      <td></td>
      <td><strong>Industrial median</strong></td>
      <td>0.4148 [0.0160]</td>
      <td>0.3361 [0.1631]</td>
      <td>-0.0320 [0.1507]</td>
      <td>0.4464 [0.0132]</td>
      <td>0.3683 [0.1477]</td>
      <td>-0.0390 [0.1094]</td>
      <td>0.2348 [0.0296]</td>
      <td>0.2388 [0.1175]</td>
      <td>-0.1703 [0.1354]</td>
    </tr>
    <tr>
      <td></td>
      <td><strong>Industrial IQR</strong></td>
      <td>0.1003 [0.0063]</td>
      <td>0.3422 [0.0680]</td>
      <td>0.1709 [0.0497]</td>
      <td>0.1056 [0.0050]</td>
      <td>0.2802 [0.0299]</td>
      <td>0.1770 [0.0335]</td>
      <td>0.2013 [0.0117]</td>
      <td>0.1632 [0.0947]</td>
      <td>0.1412 [0.1069]</td>
    </tr>
  </tbody>
</table>

CSV export: `table_1_hw_vs_llm_transfer_median_iqr.csv`.