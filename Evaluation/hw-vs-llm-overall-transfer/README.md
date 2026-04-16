# Table 1: Overall HW vs. LLM Transfer

This is the main body table for comparing how each detector transfers from
`HW / GD` inputs to `LLM-generated` inputs.

Interpretation notes:

- Each `HW` value is recomputed once over all available HW / GD samples for that detector.
- Each `LLM` value is recomputed once over all available LLM-generated samples for that detector.
- `Δ = LLM - HW`, so a negative value means the detector gets worse on LLM-generated content.
- For detector rows, `± std` is the standard deviation across available stage-level metrics.
- `mean` and `median` are computed within each detector family over detectors with available values.
- Blank cells mean the detector currently lacks a usable `HW`, `LLM`, or paired `HW -> LLM` overall result.

- `Academic` average transfer gap: `n = 6` paired detectors, `Δ Recall = -0.0721`, `Δ F2 = -0.0621`, `Δ MCC = -0.0487`.
- `Industrial` average transfer gap: `n = 3` paired detectors, `Δ Recall = 0.0357`, `Δ F2 = 0.0200`, `Δ MCC = -0.1365`.

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
      <td>scamllm</td>
      <td>0.8747 ± 0.0065</td>
      <td>0.6444 ± 0.1717</td>
      <td>-0.2302 ± 0.1726</td>
      <td>0.8633 ± 0.0311</td>
      <td>0.6659 ± 0.1360</td>
      <td>-0.1974 ± 0.1417</td>
      <td>0.6304 ± 0.0309</td>
      <td>0.3836 ± 0.2455</td>
      <td>-0.2467 ± 0.2463</td>
    </tr>
    <tr>
      <td></td>
      <td>pimref</td>
      <td>0.0408 ± 0.0036</td>
      <td>0.0167 ± 0.0206</td>
      <td>-0.0241 ± 0.0200</td>
      <td>0.0504 ± 0.0044</td>
      <td>0.0208 ± 0.0254</td>
      <td>-0.0296 ± 0.0246</td>
      <td>0.0817 ± 0.0162</td>
      <td>0.0732 ± 0.0440</td>
      <td>-0.0085 ± 0.0410</td>
    </tr>
    <tr>
      <td></td>
      <td>t5phishing</td>
      <td>1.0000 ± 0.0000</td>
      <td>1.0000 ± 0.0000</td>
      <td>0.0000 ± 0.0000</td>
      <td>0.8688 ± 0.0778</td>
      <td>0.8688 ± 0.0778</td>
      <td>0.0000 ± 0.0000</td>
      <td>0.0000 ± 0.0000</td>
      <td>0.0000 ± 0.0000</td>
      <td>0.0000 ± 0.0000</td>
    </tr>
    <tr>
      <td></td>
      <td>ml_watermark_logreg</td>
      <td>0.8587 ± 0.0072</td>
      <td>0.8418 ± 0.0807</td>
      <td>-0.0169 ± 0.0814</td>
      <td>0.7772 ± 0.0649</td>
      <td>0.7896 ± 0.0855</td>
      <td>0.0124 ± 0.0863</td>
      <td>-0.0335 ± 0.0181</td>
      <td>0.2253 ± 0.3281</td>
      <td>0.2588 ± 0.3372</td>
    </tr>
    <tr>
      <td></td>
      <td>xgboost</td>
      <td>0.6455 ± 0.0072</td>
      <td>0.6499 ± 0.1791</td>
      <td>0.0044 ± 0.1823</td>
      <td>0.6627 ± 0.0263</td>
      <td>0.6596 ± 0.1416</td>
      <td>-0.0031 ± 0.1488</td>
      <td>0.3448 ± 0.0232</td>
      <td>0.2809 ± 0.2405</td>
      <td>-0.0638 ± 0.2351</td>
    </tr>
    <tr>
      <td></td>
      <td>securenet_llama</td>
      <td>0.8312 ± 0.0110</td>
      <td>0.6652 ± 0.2705</td>
      <td>-0.1660 ± 0.2740</td>
      <td>0.8540 ± 0.0133</td>
      <td>0.6993 ± 0.2381</td>
      <td>-0.1547 ± 0.2420</td>
      <td>0.7767 ± 0.0221</td>
      <td>0.5449 ± 0.2262</td>
      <td>-0.2318 ± 0.2334</td>
    </tr>
    <tr>
      <td></td>
      <td><strong>Academic mean</strong></td>
      <td>0.7085</td>
      <td>0.6363</td>
      <td>-0.0721</td>
      <td>0.6794</td>
      <td>0.6173</td>
      <td>-0.0621</td>
      <td>0.3000</td>
      <td>0.2513</td>
      <td>-0.0487</td>
    </tr>
    <tr>
      <td></td>
      <td><strong>Academic median</strong></td>
      <td>0.8449</td>
      <td>0.6576</td>
      <td>-0.0205</td>
      <td>0.8156</td>
      <td>0.6826</td>
      <td>-0.0164</td>
      <td>0.2132</td>
      <td>0.2531</td>
      <td>-0.0362</td>
    </tr>
    <tr>
      <td><strong>Industrial</strong></td>
      <td>llm_guard</td>
      <td>0.4179 ± 0.0160</td>
      <td>0.5452 ± 0.2168</td>
      <td>0.1273 ± 0.2246</td>
      <td>0.4483 ± 0.0158</td>
      <td>0.5372 ± 0.1659</td>
      <td>0.0889 ± 0.1559</td>
      <td>0.0603 ± 0.0164</td>
      <td>-0.1581 ± 0.2647</td>
      <td>-0.2184 ± 0.2286</td>
    </tr>
    <tr>
      <td></td>
      <td>phishing_email_agent</td>
      <td>0.2158 ± 0.0107</td>
      <td>0.4035 ± 0.2145</td>
      <td>0.1877 ± 0.1830</td>
      <td>0.2536 ± 0.0095</td>
      <td>0.4495 ± 0.1848</td>
      <td>0.1959 ± 0.1235</td>
      <td>0.2185 ± 0.0337</td>
      <td>0.3224 ± 0.2217</td>
      <td>0.1039 ± 0.1533</td>
    </tr>
    <tr>
      <td></td>
      <td>email_phishing_detection_v3</td>
      <td>-</td>
      <td>0.7270 ± 0.1869</td>
      <td>-</td>
      <td>-</td>
      <td>0.7493 ± 0.1380</td>
      <td>-</td>
      <td>-</td>
      <td>0.5660 ± 0.2030</td>
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
      <td>pyrit_blocklist</td>
      <td>-</td>
      <td>0.2046 ± 0.1164</td>
      <td>-</td>
      <td>-</td>
      <td>0.2411 ± 0.1104</td>
      <td>-</td>
      <td>-</td>
      <td>0.2149 ± 0.0686</td>
      <td>-</td>
    </tr>
    <tr>
      <td></td>
      <td>spamassassin</td>
      <td>0.4125 ± 0.0206</td>
      <td>0.2047 ± 0.2245</td>
      <td>-0.2078 ± 0.2150</td>
      <td>0.4651 ± 0.0199</td>
      <td>0.2403 ± 0.1872</td>
      <td>-0.2247 ± 0.1823</td>
      <td>0.4549 ± 0.0454</td>
      <td>0.1601 ± 0.1065</td>
      <td>-0.2948 ± 0.1356</td>
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
      <td><strong>Industrial mean</strong></td>
      <td>0.3487</td>
      <td>0.4170</td>
      <td>0.0357</td>
      <td>0.3890</td>
      <td>0.4435</td>
      <td>0.0200</td>
      <td>0.2446</td>
      <td>0.2211</td>
      <td>-0.1365</td>
    </tr>
    <tr>
      <td></td>
      <td><strong>Industrial median</strong></td>
      <td>0.4125</td>
      <td>0.4035</td>
      <td>0.1273</td>
      <td>0.4483</td>
      <td>0.4495</td>
      <td>0.0889</td>
      <td>0.2185</td>
      <td>0.2149</td>
      <td>-0.2184</td>
    </tr>
  </tbody>
</table>

CSV export: `table_1_hw_vs_llm_transfer.csv`.