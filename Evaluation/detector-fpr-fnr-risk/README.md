# Detector Overall Risk Profile

This table focuses on detector risk behavior rather than aggregate task performance.

- All rates are reported as percentages and rounded to two decimals.
- `Δ = LLM - HW`, expressed in percentage points.
- Positive `Δ FPR` means more false positives on LLM-generated content.
- Positive `Δ FNR` means more false negatives on LLM-generated content.
- Blank cells mean the detector does not yet have a usable paired `HW` and `LLM` overall result.

- `Academic` mean `Δ FPR = -1.65`, `Δ FNR = 7.21`.
- `Industrial` mean `Δ FPR = 15.24`, `Δ FNR = -3.57`.

<table>
  <thead>
    <tr>
      <th rowspan="2">Group</th>
      <th rowspan="2">Detector</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Academic</strong></td>
      <td>scamllm</td>
      <td>25.29</td>
      <td>25.74</td>
      <td>0.45</td>
      <td>12.53</td>
      <td>35.56</td>
      <td>23.02</td>
    </tr>
    <tr>
      <td></td>
      <td>pimref</td>
      <td>1.32</td>
      <td>0.18</td>
      <td>-1.14</td>
      <td>95.92</td>
      <td>98.33</td>
      <td>2.41</td>
    </tr>
    <tr>
      <td></td>
      <td>t5phishing</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td></td>
      <td>ml_watermark_logreg</td>
      <td>88.16</td>
      <td>64.68</td>
      <td>-23.47</td>
      <td>14.13</td>
      <td>15.82</td>
      <td>1.69</td>
    </tr>
    <tr>
      <td></td>
      <td>xgboost</td>
      <td>29.73</td>
      <td>36.67</td>
      <td>6.93</td>
      <td>35.45</td>
      <td>35.01</td>
      <td>-0.44</td>
    </tr>
    <tr>
      <td></td>
      <td>securenet_llama</td>
      <td>4.69</td>
      <td>12.03</td>
      <td>7.33</td>
      <td>16.88</td>
      <td>33.48</td>
      <td>16.60</td>
    </tr>
    <tr>
      <td></td>
      <td><strong>Academic mean</strong></td>
      <td>41.53</td>
      <td>39.88</td>
      <td>-1.65</td>
      <td>29.15</td>
      <td>36.37</td>
      <td>7.21</td>
    </tr>
    <tr>
      <td></td>
      <td><strong>Academic median</strong></td>
      <td>27.51</td>
      <td>31.20</td>
      <td>0.23</td>
      <td>15.51</td>
      <td>34.24</td>
      <td>2.05</td>
    </tr>
    <tr>
      <td><strong>Industrial</strong></td>
      <td>llm_guard</td>
      <td>35.79</td>
      <td>70.08</td>
      <td>34.29</td>
      <td>58.21</td>
      <td>45.48</td>
      <td>-12.73</td>
    </tr>
    <tr>
      <td></td>
      <td>phishing_email_agent</td>
      <td>5.63</td>
      <td>11.18</td>
      <td>5.55</td>
      <td>78.42</td>
      <td>59.65</td>
      <td>-18.77</td>
    </tr>
    <tr>
      <td></td>
      <td>email_phishing_detection_v3</td>
      <td>-</td>
      <td>15.77</td>
      <td>-</td>
      <td>-</td>
      <td>27.30</td>
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
    </tr>
    <tr>
      <td></td>
      <td>pyrit_blocklist</td>
      <td>-</td>
      <td>5.37</td>
      <td>-</td>
      <td>-</td>
      <td>79.54</td>
      <td>-</td>
    </tr>
    <tr>
      <td></td>
      <td>spamassassin</td>
      <td>2.61</td>
      <td>8.48</td>
      <td>5.87</td>
      <td>58.75</td>
      <td>79.53</td>
      <td>20.78</td>
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
    </tr>
    <tr>
      <td></td>
      <td><strong>Industrial mean</strong></td>
      <td>14.68</td>
      <td>22.18</td>
      <td>15.24</td>
      <td>65.13</td>
      <td>58.30</td>
      <td>-3.57</td>
    </tr>
    <tr>
      <td></td>
      <td><strong>Industrial median</strong></td>
      <td>5.63</td>
      <td>11.18</td>
      <td>5.87</td>
      <td>58.75</td>
      <td>59.65</td>
      <td>-12.73</td>
    </tr>
  </tbody>
</table>

CSV export: `detector_fpr_fnr_overall.csv`.