# LaTeX Table Drafts

## Table I. Dataset Summary

```latex
\begin{table}[t]
\centering
\caption{RF-Avatar dataset summary.}
\begin{tabular}{lr}
\toprule
Statistic & Value \\
\midrule
Subjects & 12 \\
Action categories & 30 \\
Indoor scenes & 3 \\
Router deployments & 9 \\
Recording sessions & 924 \\
Valid RGB-D frames & 11331 \\
Valid CSI samples & 8009 \\
Valid BODY-38 annotations & 6987 \\
Valid point-cloud pseudo-labels & 3838 \\
Human instances & 9160 \\
Max people / frame & 3 \\
\bottomrule
\end{tabular}
\end{table}
```

## Table II. Multi-person Distribution

```latex
\begin{table}[t]
\centering
\caption{Distribution by people-count.}
\begin{tabular}{lrr}
\toprule
People count & Frames & Human instances \\
\midrule
1-person & 5238 & 5238 \\
2-person & 1325 & 2650 \\
3-person & 424 & 1272 \\
\bottomrule
\end{tabular}
\end{table}
```

## Table III. Router-position Distribution

```latex
\begin{table*}[t]
\centering
\caption{Scene and router deployment distribution.}
\begin{tabular}{llllrrr}
\toprule
Scene & Router pos. & Tx coords & Rx coords & Sessions & Frames & Human inst. \\
\midrule
lab & Pos0 & tx\_antenna\_offset\_mm:[0.0, -148.0, 34.0] & [0.0, 0.0, -78.0] & 163 & 1895 & 2744 \\
lab & Pos0\_Unseen & tx\_antenna\_offset\_mm:[0.0, -148.0, 34.0] & [0.0, 0.0, -78.0] & 107 & 0 & 0 \\
lab & Pos1\_R20 & tx\_antenna\_offset\_mm:[0.0, -148.0, 34.0] & [0.0, 0.0, -78.0] & 82 & 926 & 1480 \\
lab & Pos1\_R20\_Unseen & tx\_antenna\_offset\_mm:[0.0, -148.0, 34.0] & [0.0, 0.0, -78.0] & 79 & 0 & 0 \\
newlab & Pos2 & tx\_antenna\_offset\_mm:[0.0, -148.0, 34.0] & [0.0, 0.0, -247.0] & 243 & 1980 & 1980 \\
newlab & Pos2 & tx\_antenna\_offset\_mm:[0.0, -148.0, 34.0] & [0.0, 0.0, -379.0] & 1 & 0 & 0 \\
room & Pos0 & tx\_antenna\_offset\_mm:[0.0, -148.0, 34.0] & [0.0, 0.0, -379.0] & 138 & 1527 & 2112 \\
room & Pos0\_Unseen & tx\_antenna\_offset\_mm:[0.0, -148.0, 34.0] & [0.0, 0.0, -379.0] & 48 & 0 & 0 \\
room & Pos1\_L20 & tx\_antenna\_offset\_mm:[0.0, -148.0, 34.0] & [0.0, 0.0, -379.0] & 63 & 659 & 844 \\
\bottomrule
\end{tabular}
\end{table*}
```

## Table IV. Physical-occlusion Dataset

```latex
\begin{table}[t]
\centering
\caption{Paired physical-occlusion subset statistics.}
\begin{tabular}{lrrrrr}
\toprule
Occluder & Dist. (cm) & Sessions & Raw pairs & Valid pairs & Discarded \\
\midrule
black cloth & 100 & 28 & 401 & 255 & 146 \\
black cloth & 150 & 27 & 377 & 160 & 217 \\
black cloth & 200 & 27 & 380 & 123 & 257 \\
cardboard board & 100 & 27 & 365 & 197 & 168 \\
cardboard board & 150 & 27 & 362 & 130 & 232 \\
cardboard board & 200 & 27 & 363 & 181 & 182 \\
foam board & 100 & 27 & 364 & 348 & 16 \\
foam board & 150 & 27 & 370 & 326 & 44 \\
foam board & 200 & 27 & 362 & 260 & 102 \\
\bottomrule
\end{tabular}
\end{table}
```

## Table V. Evaluation Protocols

```latex
\begin{table}[t]
\centering
\caption{Suggested session-level evaluation protocol.}
\begin{tabular}{lrrr}
\toprule
Split & Sessions & Frames & Human instances \\
\midrule
Train & 412 & 5630 & 8091 \\
Val & 88 & 1184 & 1705 \\
Test & 91 & 1158 & 1714 \\
\bottomrule
\end{tabular}
\end{table}
```
