# Table 2: Stage-Level LLM Difficulty

This table shifts the focus from individual detectors to the stages themselves.
Each `Δ` value is computed relative to each detector's own `HW-overall` baseline:

- `Δ metric = LLM-stage metric - HW-overall metric`

Interpretation notes:

- Negative `Δ` means the stage hurts detector performance relative to that detector's HW baseline.
- `Academic` cells use the mean over available paired detectors for that stage/metric.
- `Industrial` cells currently use a fixed denominator of `n = 3` for every stage/metric.
- This version keeps `S6` and `S8` split into their sub-stages because their behavior is meaningfully different.
- Blank cells mean that detector family currently has no usable paired values for that stage/metric.

- `Academic` hardest stage by mean `Δ MCC`: `S6-fuzzer` (`-0.2306`, `n = 6`).
- `Academic` easiest stage by mean `Δ MCC`: `S6-MPG` (`0.1295`, `n = 6`).
- `Industrial` hardest stage by mean `Δ MCC`: `S2` (`-0.2926`, `n = 3`).
- `Industrial` easiest stage by mean `Δ MCC`: `S8-deepseek` (`0.0850`, `n = 3`).

<table>
  <thead>
    <tr>
      <th rowspan="3">Stage</th>
      <th colspan="9">Academic</th>
      <th colspan="9">Industrial</th>
    </tr>
    <tr>
      <th colspan="3">Recall</th>
      <th colspan="3">F2</th>
      <th colspan="3">MCC</th>
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
      <td>S1</td>
      <td>0.7085</td>
      <td>0.7893</td>
      <td>0.0808</td>
      <td>0.6794</td>
      <td>0.5776</td>
      <td>-0.1018</td>
      <td>0.3000</td>
      <td>0.2322</td>
      <td>-0.0678</td>
      <td>0.3487</td>
      <td>0.5572</td>
      <td>0.2085</td>
      <td>0.3890</td>
      <td>0.5055</td>
      <td>0.1165</td>
      <td>0.2446</td>
      <td>0.2526</td>
      <td>0.0081</td>
    </tr>
    <tr>
      <td>S2</td>
      <td>0.7085</td>
      <td>0.6958</td>
      <td>-0.0127</td>
      <td>0.6794</td>
      <td>0.6392</td>
      <td>-0.0402</td>
      <td>0.3000</td>
      <td>0.2481</td>
      <td>-0.0519</td>
      <td>0.3487</td>
      <td>0.4687</td>
      <td>0.1200</td>
      <td>0.3890</td>
      <td>0.4368</td>
      <td>0.0479</td>
      <td>0.2446</td>
      <td>-0.0480</td>
      <td>-0.2926</td>
    </tr>
    <tr>
      <td>S4</td>
      <td>0.7085</td>
      <td>0.5983</td>
      <td>-0.1102</td>
      <td>0.6794</td>
      <td>0.6056</td>
      <td>-0.0738</td>
      <td>0.3000</td>
      <td>0.2898</td>
      <td>-0.0102</td>
      <td>0.3487</td>
      <td>0.3824</td>
      <td>0.0337</td>
      <td>0.3890</td>
      <td>0.4130</td>
      <td>0.0240</td>
      <td>0.2446</td>
      <td>0.0662</td>
      <td>-0.1784</td>
    </tr>
    <tr>
      <td>S5</td>
      <td>0.7085</td>
      <td>0.7814</td>
      <td>0.0729</td>
      <td>0.6794</td>
      <td>0.7234</td>
      <td>0.0440</td>
      <td>0.3000</td>
      <td>0.3948</td>
      <td>0.0948</td>
      <td>0.2112</td>
      <td>0.4593</td>
      <td>0.2481</td>
      <td>0.2339</td>
      <td>0.4507</td>
      <td>0.2168</td>
      <td>0.0929</td>
      <td>0.0984</td>
      <td>0.0055</td>
    </tr>
    <tr>
      <td>S6-MPG</td>
      <td>0.7085</td>
      <td>0.7595</td>
      <td>0.0510</td>
      <td>0.6794</td>
      <td>0.7151</td>
      <td>0.0357</td>
      <td>0.3000</td>
      <td>0.4295</td>
      <td>0.1295</td>
      <td>0.2112</td>
      <td>0.2380</td>
      <td>0.0268</td>
      <td>0.2339</td>
      <td>0.2595</td>
      <td>0.0256</td>
      <td>0.0929</td>
      <td>0.1097</td>
      <td>0.0168</td>
    </tr>
    <tr>
      <td>S6-UTA</td>
      <td>0.7085</td>
      <td>0.6346</td>
      <td>-0.0739</td>
      <td>0.6794</td>
      <td>0.6013</td>
      <td>-0.0781</td>
      <td>0.3000</td>
      <td>0.2830</td>
      <td>-0.0170</td>
      <td>0.2112</td>
      <td>0.2098</td>
      <td>-0.0014</td>
      <td>0.2339</td>
      <td>0.2300</td>
      <td>-0.0039</td>
      <td>0.0929</td>
      <td>0.0578</td>
      <td>-0.0352</td>
    </tr>
    <tr>
      <td>S6-fuzzer</td>
      <td>0.7085</td>
      <td>0.6397</td>
      <td>-0.0687</td>
      <td>0.6794</td>
      <td>0.6245</td>
      <td>-0.0549</td>
      <td>0.3000</td>
      <td>0.0694</td>
      <td>-0.2306</td>
      <td>0.2112</td>
      <td>0.2148</td>
      <td>0.0035</td>
      <td>0.2339</td>
      <td>0.2350</td>
      <td>0.0010</td>
      <td>0.0929</td>
      <td>-0.0634</td>
      <td>-0.1564</td>
    </tr>
    <tr>
      <td>S8-deepseek</td>
      <td>0.7085</td>
      <td>0.5589</td>
      <td>-0.1496</td>
      <td>0.6794</td>
      <td>0.5038</td>
      <td>-0.1756</td>
      <td>0.3000</td>
      <td>0.1346</td>
      <td>-0.1654</td>
      <td>0.2112</td>
      <td>0.3680</td>
      <td>0.1568</td>
      <td>0.2339</td>
      <td>0.3547</td>
      <td>0.1207</td>
      <td>0.0929</td>
      <td>0.1779</td>
      <td>0.0850</td>
    </tr>
    <tr>
      <td>S8-llama</td>
      <td>0.7085</td>
      <td>0.6481</td>
      <td>-0.0604</td>
      <td>0.6794</td>
      <td>0.6138</td>
      <td>-0.0656</td>
      <td>0.3000</td>
      <td>0.2645</td>
      <td>-0.0355</td>
      <td>0.2112</td>
      <td>0.3713</td>
      <td>0.1601</td>
      <td>0.2339</td>
      <td>0.3665</td>
      <td>0.1325</td>
      <td>0.0929</td>
      <td>0.1107</td>
      <td>0.0177</td>
    </tr>
    <tr>
      <td>S8-ministral</td>
      <td>0.7085</td>
      <td>0.6384</td>
      <td>-0.0700</td>
      <td>0.6794</td>
      <td>0.6084</td>
      <td>-0.0710</td>
      <td>0.3000</td>
      <td>0.2609</td>
      <td>-0.0391</td>
      <td>0.2112</td>
      <td>0.3171</td>
      <td>0.1059</td>
      <td>0.2339</td>
      <td>0.3182</td>
      <td>0.0842</td>
      <td>0.0929</td>
      <td>0.1176</td>
      <td>0.0246</td>
    </tr>
  </tbody>
</table>

CSV export: `table_2_stage_transfer_trend.csv`.