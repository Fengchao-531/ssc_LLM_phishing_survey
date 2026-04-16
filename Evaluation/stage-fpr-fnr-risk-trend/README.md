# Stage-Level FPR/FNR Risk Trend

This table shows how stage difficulty changes detector risk behavior.

- All rates are reported as percentages and rounded to two decimals.
- `Δ = LLM-stage - HW-overall`, expressed in percentage points.
- Positive `Δ FPR` means more false positives on that LLM stage.
- Positive `Δ FNR` means more false negatives on that LLM stage.
- `Academic` cells use the mean over available paired detectors for that stage/metric.
- `Industrial` cells currently use a fixed denominator of `n = 3` for every stage/metric.

- `Academic` largest `Δ FPR`: `S6-fuzzer` (`15.45`, `n = 6`).
- `Academic` largest `Δ FNR`: `S8-deepseek` (`14.96`, `n = 6`).
- `Industrial` largest `Δ FPR`: `S2` (`33.62`, `n = 3`).
- `Industrial` largest `Δ FNR`: `S6-UTA` (`0.14`, `n = 3`).

<table>
  <thead>
    <tr>
      <th rowspan="3">Stage</th>
      <th colspan="6">Academic</th>
      <th colspan="6">Industrial</th>
    </tr>
    <tr>
      <th colspan="3">FPR</th>
      <th colspan="3">FNR</th>
      <th colspan="3">FPR</th>
      <th colspan="3">FNR</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>S1</td>
      <td>41.53</td>
      <td>53.99</td>
      <td>12.46</td>
      <td>29.15</td>
      <td>21.07</td>
      <td>-8.08</td>
      <td>14.68</td>
      <td>28.30</td>
      <td>13.62</td>
      <td>65.13</td>
      <td>44.28</td>
      <td>-20.85</td>
    </tr>
    <tr>
      <td>S2</td>
      <td>41.53</td>
      <td>47.56</td>
      <td>6.03</td>
      <td>29.15</td>
      <td>30.42</td>
      <td>1.27</td>
      <td>14.68</td>
      <td>48.30</td>
      <td>33.62</td>
      <td>65.13</td>
      <td>53.13</td>
      <td>-12.00</td>
    </tr>
    <tr>
      <td>S4</td>
      <td>41.53</td>
      <td>30.07</td>
      <td>-11.46</td>
      <td>29.15</td>
      <td>40.17</td>
      <td>11.02</td>
      <td>14.68</td>
      <td>34.09</td>
      <td>19.42</td>
      <td>65.13</td>
      <td>61.76</td>
      <td>-3.37</td>
    </tr>
    <tr>
      <td>S5</td>
      <td>41.53</td>
      <td>39.14</td>
      <td>-2.39</td>
      <td>29.15</td>
      <td>21.86</td>
      <td>-7.29</td>
      <td>13.81</td>
      <td>33.55</td>
      <td>19.74</td>
      <td>45.54</td>
      <td>20.74</td>
      <td>-24.81</td>
    </tr>
    <tr>
      <td>S6-MPG</td>
      <td>41.53</td>
      <td>35.21</td>
      <td>-6.32</td>
      <td>29.15</td>
      <td>24.05</td>
      <td>-5.10</td>
      <td>13.81</td>
      <td>14.61</td>
      <td>0.80</td>
      <td>45.54</td>
      <td>42.86</td>
      <td>-2.68</td>
    </tr>
    <tr>
      <td>S6-UTA</td>
      <td>41.53</td>
      <td>37.65</td>
      <td>-3.89</td>
      <td>29.15</td>
      <td>36.54</td>
      <td>7.39</td>
      <td>13.81</td>
      <td>16.25</td>
      <td>2.44</td>
      <td>45.54</td>
      <td>45.69</td>
      <td>0.14</td>
    </tr>
    <tr>
      <td>S6-fuzzer</td>
      <td>41.53</td>
      <td>56.98</td>
      <td>15.45</td>
      <td>29.15</td>
      <td>36.03</td>
      <td>6.87</td>
      <td>13.81</td>
      <td>29.01</td>
      <td>15.21</td>
      <td>45.54</td>
      <td>45.19</td>
      <td>-0.35</td>
    </tr>
    <tr>
      <td>S8-deepseek</td>
      <td>41.53</td>
      <td>46.36</td>
      <td>4.82</td>
      <td>29.15</td>
      <td>44.11</td>
      <td>14.96</td>
      <td>13.81</td>
      <td>23.27</td>
      <td>9.46</td>
      <td>45.54</td>
      <td>29.87</td>
      <td>-15.68</td>
    </tr>
    <tr>
      <td>S8-llama</td>
      <td>41.53</td>
      <td>40.70</td>
      <td>-0.83</td>
      <td>29.15</td>
      <td>35.19</td>
      <td>6.04</td>
      <td>13.81</td>
      <td>27.56</td>
      <td>13.75</td>
      <td>45.54</td>
      <td>29.53</td>
      <td>-16.01</td>
    </tr>
    <tr>
      <td>S8-ministral</td>
      <td>41.53</td>
      <td>40.13</td>
      <td>-1.40</td>
      <td>29.15</td>
      <td>36.16</td>
      <td>7.00</td>
      <td>13.81</td>
      <td>22.84</td>
      <td>9.04</td>
      <td>45.54</td>
      <td>34.96</td>
      <td>-10.59</td>
    </tr>
  </tbody>
</table>

CSV export: `stage_fpr_fnr_trend.csv`.